import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import KDTree
from torch import nn

from colmap_reader import SceneInfo
from sh_rgb import RGB2SH, eval_sh


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    matrix = torch.stack(
        [
            ww + xx - yy - zz,
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            ww - xx + yy - zz,
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    )
    return matrix.view(*quaternion.shape[:-1], 3, 3)


def build_covariance_from_scaling_rotation(
    scaling: torch.Tensor,
    scaling_modifier: float,
    rotation: torch.Tensor,
) -> torch.Tensor:
    scaled = scaling * scaling_modifier
    rotation_matrix = quaternion_to_matrix(rotation)
    diag = torch.diag_embed(scaled)
    transform = rotation_matrix @ diag
    return transform @ transform.transpose(-1, -2)


@dataclass
class OptimConfig:
    position_lr_init: float = 1.6e-4
    feature_lr: float = 2.5e-3
    opacity_lr: float = 5.0e-2
    scaling_lr: float = 5.0e-3
    rotation_lr: float = 1.0e-3


class GaussianModel(nn.Module):
    def __init__(self, scene: SceneInfo, sh_degree: int = 3, max_points: int | None = None, seed: int = 0):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.spatial_lr_scale = float(scene.nerf_normalization["radius"])

        points = torch.as_tensor(scene.point_cloud.points, dtype=torch.float32)
        colors = torch.as_tensor(scene.point_cloud.colors, dtype=torch.float32)

        if max_points is not None and max_points < points.shape[0]:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            perm = torch.randperm(points.shape[0], generator=generator)[:max_points]
            points = points[perm]
            colors = colors[perm]

        num_points = points.shape[0]
        sh_dim = (self.max_sh_degree + 1) ** 2

        features = torch.zeros((num_points, sh_dim, 3), dtype=torch.float32)
        features[:, 0, :] = RGB2SH(colors)

        dist2 = torch.clamp_min(self.dist_kdtree(points.cpu().numpy()).float(), 1e-7)
        scaling = 0.5 * torch.log(dist2).unsqueeze(-1).repeat(1, 3)

        rotation = torch.zeros((num_points, 4), dtype=torch.float32)
        rotation[:, 0] = 1.0

        opacity = inverse_sigmoid(torch.full((num_points, 1), 0.25, dtype=torch.float32))

        self.xyz = nn.Parameter(points)
        self.features_dc = nn.Parameter(features[:, :1, :].contiguous())
        self.features_rest = nn.Parameter(features[:, 1:, :].contiguous())
        self.scaling_logits = nn.Parameter(scaling)
        self.rotation = nn.Parameter(rotation)
        self.opacity_logits = nn.Parameter(opacity)

    @property
    def scaling(self) -> torch.Tensor:
        return torch.exp(self.scaling_logits)

    @property
    def features(self) -> torch.Tensor:
        return torch.cat((self.features_dc, self.features_rest), dim=1)

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    @property
    def num_points(self) -> int:
        return int(self.xyz.shape[0])

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        return build_covariance_from_scaling_rotation(self.scaling, scaling_modifier, self.rotation)

    def get_colors(self, view_dirs: torch.Tensor) -> torch.Tensor:
        sh_coeff = self.features.transpose(1, 2)
        rgb = eval_sh(self.active_sh_degree, sh_coeff, view_dirs)
        return torch.clamp(rgb + 0.5, 0.0, 1.0)

    def oneupSHdegree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args) -> torch.optim.Optimizer:
        if isinstance(training_args, dict):
            cfg = OptimConfig(**training_args)
        else:
            cfg = OptimConfig(
                position_lr_init=getattr(training_args, "position_lr_init", OptimConfig.position_lr_init),
                feature_lr=getattr(training_args, "feature_lr", OptimConfig.feature_lr),
                opacity_lr=getattr(training_args, "opacity_lr", OptimConfig.opacity_lr),
                scaling_lr=getattr(training_args, "scaling_lr", OptimConfig.scaling_lr),
                rotation_lr=getattr(training_args, "rotation_lr", OptimConfig.rotation_lr),
            )

        param_groups = [
            {"params": [self.xyz], "lr": cfg.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self.features_dc], "lr": cfg.feature_lr, "name": "f_dc"},
            {"params": [self.features_rest], "lr": cfg.feature_lr * 0.25, "name": "f_rest"},
            {"params": [self.opacity_logits], "lr": cfg.opacity_lr, "name": "opacity"},
            {"params": [self.scaling_logits], "lr": cfg.scaling_lr, "name": "scaling"},
            {"params": [self.rotation], "lr": cfg.rotation_lr, "name": "rotation"},
        ]
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def dist_kdtree(self, points_np: np.ndarray) -> torch.Tensor:
        dists, _ = KDTree(points_np).query(points_np, k=min(4, points_np.shape[0]))
        if dists.ndim == 1:
            dists = dists[:, None]
        if dists.shape[1] == 1:
            mean_dists = np.full((points_np.shape[0],), 1e-4, dtype=np.float32)
        else:
            mean_dists = (dists[:, 1:] ** 2).mean(axis=1)
        return torch.from_numpy(mean_dists.astype(np.float32))

    def _set_parameters(
        self,
        xyz: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        scaling_logits: torch.Tensor,
        rotation: torch.Tensor,
        opacity_logits: torch.Tensor,
    ) -> None:
        self.xyz = nn.Parameter(xyz.contiguous())
        self.features_dc = nn.Parameter(features_dc.contiguous())
        self.features_rest = nn.Parameter(features_rest.contiguous())
        self.scaling_logits = nn.Parameter(scaling_logits.contiguous())
        self.rotation = nn.Parameter(rotation.contiguous())
        self.opacity_logits = nn.Parameter(opacity_logits.contiguous())

    def append_points(
        self,
        xyz: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        scaling_logits: torch.Tensor,
        rotation: torch.Tensor,
        opacity_logits: torch.Tensor,
    ) -> None:
        self._set_parameters(
            torch.cat([self.xyz.detach(), xyz], dim=0),
            torch.cat([self.features_dc.detach(), features_dc], dim=0),
            torch.cat([self.features_rest.detach(), features_rest], dim=0),
            torch.cat([self.scaling_logits.detach(), scaling_logits], dim=0),
            torch.cat([self.rotation.detach(), rotation], dim=0),
            torch.cat([self.opacity_logits.detach(), opacity_logits], dim=0),
        )

    def prune_points(self, keep_mask: torch.Tensor) -> bool:
        keep_mask = keep_mask.to(device=self.xyz.device, dtype=torch.bool)
        if keep_mask.all():
            return False
        self._set_parameters(
            self.xyz.detach()[keep_mask],
            self.features_dc.detach()[keep_mask],
            self.features_rest.detach()[keep_mask],
            self.scaling_logits.detach()[keep_mask],
            self.rotation.detach()[keep_mask],
            self.opacity_logits.detach()[keep_mask],
        )
        return True

    def clone_points(self, indices: torch.Tensor, clone_factor: int = 2, jitter_scale: float = 0.35) -> bool:
        indices = indices.to(device=self.xyz.device, dtype=torch.long)
        if indices.numel() == 0:
            return False

        repeated = indices.repeat_interleave(clone_factor)
        scales = self.scaling.detach()[repeated]
        noise = torch.randn_like(scales) * scales * jitter_scale
        new_xyz = self.xyz.detach()[repeated] + noise
        new_features_dc = self.features_dc.detach()[repeated]
        new_features_rest = self.features_rest.detach()[repeated]
        new_scaling_logits = self.scaling_logits.detach()[repeated] + math.log(0.8)
        new_rotation = self.rotation.detach()[repeated]
        new_opacity = (self.opacity.detach()[repeated] * 0.5).clamp(min=1e-3, max=0.95)
        new_opacity_logits = inverse_sigmoid(new_opacity)

        parent_opacity = self.opacity.detach().clone()
        parent_opacity[indices] = (parent_opacity[indices] * 0.75).clamp(min=1e-3, max=0.95)
        self.opacity_logits = nn.Parameter(inverse_sigmoid(parent_opacity))

        self.append_points(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_scaling_logits,
            new_rotation,
            new_opacity_logits,
        )
        return True

    def densify_and_prune(
        self,
        xyz_grad: torch.Tensor,
        grad_threshold: float,
        min_opacity: float,
        max_scale: float,
        max_points: int,
        clone_factor: int = 2,
        jitter_scale: float = 0.35,
    ) -> bool:
        changed = False
        device = self.xyz.device
        grad_norm = xyz_grad.to(device).norm(dim=-1)

        opacity = self.opacity.detach().squeeze(-1)
        scales = self.scaling.detach().max(dim=-1).values
        keep_mask = (opacity >= min_opacity) & (scales <= max_scale)
        if keep_mask.sum().item() >= 32:
            changed = self.prune_points(keep_mask) or changed
            grad_norm = grad_norm[keep_mask]
            opacity = opacity[keep_mask]
            scales = scales[keep_mask]

        capacity = max(0, max_points - self.num_points)
        if capacity == 0:
            return changed

        candidate_mask = grad_norm >= grad_threshold
        candidate_mask &= opacity >= max(min_opacity * 1.5, 0.05)
        candidate_mask &= scales <= max_scale * 0.75
        candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        if candidate_idx.numel() == 0:
            return changed

        max_seed_points = max(1, capacity // max(clone_factor, 1))
        seed_count = min(candidate_idx.numel(), max_seed_points)
        topk = torch.topk(grad_norm[candidate_idx], k=seed_count, largest=True).indices
        selected = candidate_idx[topk]
        changed = self.clone_points(selected, clone_factor=clone_factor, jitter_scale=jitter_scale) or changed
        return changed

    def capture(self) -> dict:
        return {
            "state_dict": self.state_dict(),
            "active_sh_degree": self.active_sh_degree,
            "max_sh_degree": self.max_sh_degree,
            "num_points": self.num_points,
        }

    def restore(self, payload: dict) -> None:
        self.load_state_dict(payload["state_dict"])
        self.active_sh_degree = int(payload.get("active_sh_degree", 0))
        self.max_sh_degree = int(payload.get("max_sh_degree", self.max_sh_degree))


if __name__ == "__main__":
    from colmap_reader import read_colmap_scene_info

    scene = read_colmap_scene_info("data/playroom", image_scale=8)
    model = GaussianModel(scene, max_points=2048)
    print("Number of Gaussians:", model.num_points)
    print("Covariance shape:", model.get_covariance().shape)
