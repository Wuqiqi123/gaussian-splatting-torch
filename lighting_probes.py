"""SH-based lighting probes for 3D Gaussian Splatting.

Each probe stores a learnable [sh_dim, 3] SH coefficient tensor.
During rendering, each Gaussian blends the K nearest probes (distance-weighted)
and the result is added directly to the Gaussian's own SH coefficients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightingProbes(nn.Module):
    """Learnable SH lighting probes placed on a uniform grid in the scene AABB.

    Args:
        scene_aabb:  [2, 3] tensor — (min_xyz, max_xyz) of the scene
        grid_size:   number of probes per axis (total = grid_size^3)
        sh_degree:   max SH degree (sh_dim = (sh_degree+1)^2)
        k_nearest:   number of nearest probes to blend per Gaussian
    """

    def __init__(
        self,
        scene_aabb: torch.Tensor,
        grid_size: int = 3,
        sh_degree: int = 3,
        k_nearest: int = 4,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.sh_degree = sh_degree
        self.k_nearest = k_nearest

        lo, hi = scene_aabb[0], scene_aabb[1]
        coords = [torch.linspace(float(lo[i]), float(hi[i]), grid_size) for i in range(3)]
        grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)
        positions = grid.reshape(-1, 3)  # [M, 3]
        self.register_buffer("probe_positions", positions)

        M = positions.shape[0]
        sh_dim = (sh_degree + 1) ** 2
        self.sh_coeffs = nn.Parameter(torch.zeros(M, sh_dim, 3))

    @property
    def num_probes(self) -> int:
        return int(self.probe_positions.shape[0])

    def query(self, xyz: torch.Tensor, active_sh_degree: int) -> torch.Tensor:
        """Return blended SH coefficient delta for each Gaussian.

        Args:
            xyz:              [N, 3] Gaussian world positions
            active_sh_degree: current active SH degree (bands 0..active_sh_degree)

        Returns:
            sh_delta: [N, sh_dim, 3] — same shape as gaussian_model.features,
                      with coefficients beyond active_sh_degree zeroed out.
        """
        N = xyz.shape[0]
        K = min(self.k_nearest, self.num_probes)
        sh_dim = (self.sh_degree + 1) ** 2
        active_dim = (active_sh_degree + 1) ** 2

        dists = torch.cdist(xyz, self.probe_positions)           # [N, M]
        topk_dists, topk_idx = dists.topk(K, dim=-1, largest=False)  # [N, K]

        weights = 1.0 / (topk_dists + 1e-4)
        weights = weights / weights.sum(dim=-1, keepdim=True)    # [N, K]

        # Gather probe SH coeffs: [N, K, sh_dim, 3]
        probe_sh = self.sh_coeffs[topk_idx]                      # [N, K, sh_dim, 3]
        # Weighted sum over K probes
        sh_delta = (probe_sh * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [N, sh_dim, 3]

        # Zero out bands beyond active degree
        if active_dim < sh_dim:
            sh_delta = sh_delta.clone()
            sh_delta[:, active_dim:, :] = 0.0

        return sh_delta

    def capture(self) -> dict:
        return {
            "sh_coeffs": self.sh_coeffs.data.clone(),
            "probe_positions": self.probe_positions.clone(),
            "grid_size": self.grid_size,
            "sh_degree": self.sh_degree,
            "k_nearest": self.k_nearest,
        }

    def restore(self, payload: dict) -> None:
        self.sh_coeffs = nn.Parameter(payload["sh_coeffs"].to(self.sh_coeffs.device))
        self.probe_positions = payload["probe_positions"].to(self.probe_positions.device)
        self.grid_size = int(payload["grid_size"])
        self.sh_degree = int(payload["sh_degree"])
        self.k_nearest = int(payload["k_nearest"])


def build_probes_from_scene(
    points: torch.Tensor,
    grid_size: int = 3,
    sh_degree: int = 3,
    k_nearest: int = 4,
    padding: float = 0.1,
) -> LightingProbes:
    lo = points.min(dim=0).values
    hi = points.max(dim=0).values
    pad = (hi - lo) * padding
    aabb = torch.stack([lo - pad, hi + pad], dim=0)
    return LightingProbes(aabb, grid_size=grid_size, sh_degree=sh_degree, k_nearest=k_nearest)
