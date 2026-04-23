"""Cubemap-based lighting probes for 3D Gaussian Splatting.

Each probe stores a learnable [6, H, W, 3] cubemap. During rendering,
each Gaussian queries the K nearest probes (distance-weighted) using its
view direction, and the result is added on top of the SH color.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dir_to_face_uv(dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert unit direction vectors to cubemap face index and UV in [-1, 1].

    Args:
        dirs: [N, 3] unit direction vectors (x, y, z)

    Returns:
        face: [N] long tensor, face index 0-5 (+X,-X,+Y,-Y,+Z,-Z)
        uv:   [N, 2] float tensor in [-1, 1]
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    ax, ay, az = x.abs(), y.abs(), z.abs()

    # Determine dominant axis → face index
    # face 0: +X (ax >= ay, ax >= az, x > 0)
    # face 1: -X (ax >= ay, ax >= az, x <= 0)
    # face 2: +Y (ay > ax, ay >= az, y > 0)
    # face 3: -Y (ay > ax, ay >= az, y <= 0)
    # face 4: +Z (az > ax, az > ay, z > 0)
    # face 5: -Z (az > ax, az > ay, z <= 0)
    x_dom = (ax >= ay) & (ax >= az)
    y_dom = (ay > ax) & (ay >= az)
    # z_dom = ~x_dom & ~y_dom  (implicit)

    face = torch.zeros(dirs.shape[0], dtype=torch.long, device=dirs.device)
    face[x_dom & (x > 0)] = 0
    face[x_dom & (x <= 0)] = 1
    face[y_dom & (y > 0)] = 2
    face[y_dom & (y <= 0)] = 3
    face[(~x_dom & ~y_dom) & (z > 0)] = 4
    face[(~x_dom & ~y_dom) & (z <= 0)] = 5

    # Compute UV for each face
    # Convention: u = right axis / max_axis, v = up axis / max_axis
    # Face 0 (+X): sc=z, tc=y, ma=x  → u=-z/x, v=-y/x
    # Face 1 (-X): sc=-z, tc=y, ma=-x → u=z/(-x), v=-y/(-x)
    # Face 2 (+Y): sc=x, tc=-z, ma=y  → u=x/y, v=z/y
    # Face 3 (-Y): sc=x, tc=z, ma=-y  → u=x/(-y), v=-z/(-y)
    # Face 4 (+Z): sc=x, tc=y, ma=z   → u=x/z, v=-y/z
    # Face 5 (-Z): sc=-x, tc=y, ma=-z → u=-x/(-z), v=-y/(-z)
    eps = 1e-8
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)

    m = face == 0; u[m] = -z[m] / (ax[m] + eps); v[m] = -y[m] / (ax[m] + eps)
    m = face == 1; u[m] =  z[m] / (ax[m] + eps); v[m] = -y[m] / (ax[m] + eps)
    m = face == 2; u[m] =  x[m] / (ay[m] + eps); v[m] =  z[m] / (ay[m] + eps)
    m = face == 3; u[m] =  x[m] / (ay[m] + eps); v[m] = -z[m] / (ay[m] + eps)
    m = face == 4; u[m] =  x[m] / (az[m] + eps); v[m] = -y[m] / (az[m] + eps)
    m = face == 5; u[m] = -x[m] / (az[m] + eps); v[m] = -y[m] / (az[m] + eps)

    uv = torch.stack([u, v], dim=-1).clamp(-1.0, 1.0)
    return face, uv


def _sample_cubemap(cubemaps: torch.Tensor, probe_idx: torch.Tensor,
                    face: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Sample colors from cubemaps for a batch of (probe, face, uv) queries.

    Args:
        cubemaps:  [M, 6, H, W, 3] learnable cubemap tensor
        probe_idx: [N] long — which probe to sample for each query
        face:      [N] long — which face (0-5)
        uv:        [N, 2] float in [-1, 1]

    Returns:
        colors: [N, 3]
    """
    M, num_faces, H, W, C = cubemaps.shape
    N = probe_idx.shape[0]

    # cubemaps[probe_idx, face]: select the right face image per query → [N, H, W, 3]
    # Reshape to [N, C, H, W] for grid_sample
    face_imgs = cubemaps[probe_idx, face]          # [N, H, W, 3]
    face_imgs = face_imgs.permute(0, 3, 1, 2)      # [N, 3, H, W]

    # grid_sample expects grid [N, 1, 1, 2]
    grid = uv.view(N, 1, 1, 2)                     # [N, 1, 1, 2]
    sampled = F.grid_sample(
        face_imgs, grid,
        mode="bilinear", padding_mode="border", align_corners=True
    )  # [N, 3, 1, 1]
    return sampled.view(N, C)                       # [N, 3]


class LightingProbes(nn.Module):
    """Learnable cubemap lighting probes placed on a uniform grid in the scene AABB.

    Args:
        scene_aabb:  [2, 3] tensor — (min_xyz, max_xyz) of the scene
        grid_size:   number of probes per axis (total = grid_size^3)
        cubemap_res: resolution H=W of each cubemap face
        k_nearest:   number of nearest probes to blend per Gaussian
    """

    def __init__(
        self,
        scene_aabb: torch.Tensor,
        grid_size: int = 3,
        cubemap_res: int = 8,
        k_nearest: int = 4,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.cubemap_res = cubemap_res
        self.k_nearest = k_nearest

        # Build probe positions on a uniform grid
        lo, hi = scene_aabb[0], scene_aabb[1]
        coords = [torch.linspace(float(lo[i]), float(hi[i]), grid_size) for i in range(3)]
        grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)  # [G,G,G,3]
        positions = grid.reshape(-1, 3)  # [M, 3]
        self.register_buffer("probe_positions", positions)

        M = positions.shape[0]
        # Initialize cubemaps to small random values (near-zero → probes start neutral)
        self.cubemaps = nn.Parameter(
            torch.zeros(M, 6, cubemap_res, cubemap_res, 3)
        )

    @property
    def num_probes(self) -> int:
        return int(self.probe_positions.shape[0])

    def query(self, xyz: torch.Tensor, view_dirs: torch.Tensor) -> torch.Tensor:
        """Query probe colors for a set of Gaussians.

        Args:
            xyz:       [N, 3] Gaussian world positions
            view_dirs: [N, 3] unit view directions (camera → Gaussian, or Gaussian → camera)

        Returns:
            colors: [N, 3] probe contribution (to be added to SH color)
        """
        N = xyz.shape[0]
        K = min(self.k_nearest, self.num_probes)

        # --- Distance-weighted probe selection ---
        # dists: [N, M]
        dists = torch.cdist(xyz, self.probe_positions)          # [N, M]
        topk_dists, topk_idx = dists.topk(K, dim=-1, largest=False)  # [N, K]

        # Inverse-distance weights, softmax-normalized
        weights = 1.0 / (topk_dists + 1e-4)                    # [N, K]
        weights = weights / weights.sum(dim=-1, keepdim=True)   # [N, K]

        # --- Cubemap lookup per (probe, Gaussian) pair ---
        # Normalize view directions
        dirs = F.normalize(view_dirs, dim=-1)                   # [N, 3]
        face, uv = _dir_to_face_uv(dirs)                        # [N], [N, 2]

        # Expand for K probes: repeat each Gaussian K times
        dirs_rep = dirs.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)
        face_rep = face.unsqueeze(1).expand(-1, K).reshape(N * K)
        uv_rep   = uv.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 2)
        probe_idx_flat = topk_idx.reshape(N * K)                # [N*K]

        sampled = _sample_cubemap(
            self.cubemaps, probe_idx_flat, face_rep, uv_rep
        )  # [N*K, 3]

        sampled = sampled.view(N, K, 3)                         # [N, K, 3]
        colors = (sampled * weights.unsqueeze(-1)).sum(dim=1)   # [N, 3]
        return colors

    def capture(self) -> dict:
        return {
            "cubemaps": self.cubemaps.data.clone(),
            "probe_positions": self.probe_positions.clone(),
            "grid_size": self.grid_size,
            "cubemap_res": self.cubemap_res,
            "k_nearest": self.k_nearest,
        }

    def restore(self, payload: dict) -> None:
        self.cubemaps = nn.Parameter(payload["cubemaps"].to(self.cubemaps.device))
        self.probe_positions = payload["probe_positions"].to(self.probe_positions.device)
        self.grid_size = int(payload["grid_size"])
        self.cubemap_res = int(payload["cubemap_res"])
        self.k_nearest = int(payload["k_nearest"])


def build_probes_from_scene(
    points: torch.Tensor,
    grid_size: int = 3,
    cubemap_res: int = 8,
    k_nearest: int = 4,
    padding: float = 0.1,
) -> LightingProbes:
    """Construct LightingProbes from a point cloud, computing AABB automatically.

    Args:
        points:     [N, 3] scene point cloud
        grid_size:  probes per axis
        cubemap_res: cubemap face resolution
        k_nearest:  probes to blend per Gaussian
        padding:    fractional padding around AABB
    """
    lo = points.min(dim=0).values
    hi = points.max(dim=0).values
    pad = (hi - lo) * padding
    aabb = torch.stack([lo - pad, hi + pad], dim=0)  # [2, 3]
    return LightingProbes(aabb, grid_size=grid_size, cubemap_res=cubemap_res, k_nearest=k_nearest)
