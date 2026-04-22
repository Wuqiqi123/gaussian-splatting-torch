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
        self.register_buffer("xyz_gradient_accum", torch.zeros((0,), dtype=torch.float32), persistent=False)
        self.register_buffer("denom", torch.zeros((0,), dtype=torch.float32), persistent=False)
        self.register_buffer("max_radii2D", torch.zeros((0,), dtype=torch.float32), persistent=False)

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
        self._reset_density_state()

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

    def _reset_density_state(self) -> None:
        device = self.xyz.device
        self.xyz_gradient_accum = torch.zeros((self.num_points,), device=device, dtype=torch.float32)
        self.denom = torch.zeros((self.num_points,), device=device, dtype=torch.float32)
        self.max_radii2D = torch.zeros((self.num_points,), device=device, dtype=torch.float32)

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
        self._reset_density_state()

    def add_densification_stats(
        self,
        viewspace_points_grad: torch.Tensor | None,
        visible_indices: torch.Tensor,
        radii: torch.Tensor | None = None,
    ) -> None:
        if viewspace_points_grad is None or visible_indices.numel() == 0:
            return
        visible_indices = visible_indices.to(device=self.xyz.device, dtype=torch.long)
        grad_norm = torch.norm(viewspace_points_grad[visible_indices, :2], dim=-1)
        grad_norm = torch.nan_to_num(grad_norm, nan=0.0, posinf=0.0, neginf=0.0)
        self.xyz_gradient_accum[visible_indices] += grad_norm
        self.denom[visible_indices] += 1.0
        if radii is not None:
            visible_radii = radii[visible_indices].to(self.max_radii2D)
            self.max_radii2D[visible_indices] = torch.maximum(self.max_radii2D[visible_indices], visible_radii)

    def reset_opacity(self, max_opacity: float = 0.01) -> None:
        new_opacity = torch.minimum(self.opacity.detach(), torch.full_like(self.opacity.detach(), max_opacity))
        self._set_parameters(
            self.xyz.detach(),
            self.features_dc.detach(),
            self.features_rest.detach(),
            self.scaling_logits.detach(),
            self.rotation.detach(),
            inverse_sigmoid(new_opacity),
        )

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
        local_noise = torch.randn_like(scales) * scales * jitter_scale
        rotation = quaternion_to_matrix(self.rotation.detach()[repeated])
        new_xyz = self.xyz.detach()[repeated] + torch.einsum("nij,nj->ni", rotation, local_noise)
        new_features_dc = self.features_dc.detach()[repeated]
        new_features_rest = self.features_rest.detach()[repeated]
        new_scaling_logits = self.scaling_logits.detach()[repeated] - math.log(1.6)
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

    def _split_points(
        self,
        indices: torch.Tensor,
        split_factor: int = 2,
        scale_shrink: float = 1.6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        repeated = indices.repeat_interleave(split_factor)
        scales = self.scaling.detach()[repeated]
        rotation = quaternion_to_matrix(self.rotation.detach()[repeated])
        local_samples = torch.randn_like(scales) * scales
        new_xyz = self.xyz.detach()[repeated] + torch.einsum("nij,nj->ni", rotation, local_samples)
        new_features_dc = self.features_dc.detach()[repeated]
        new_features_rest = self.features_rest.detach()[repeated]
        new_scaling_logits = self.scaling_logits.detach()[repeated] - math.log(scale_shrink)
        new_rotation = self.rotation.detach()[repeated]
        new_opacity = (self.opacity.detach()[repeated] / float(split_factor)).clamp(min=1e-3, max=0.95)
        new_opacity_logits = inverse_sigmoid(new_opacity)
        return new_xyz, new_features_dc, new_features_rest, new_scaling_logits, new_rotation, new_opacity_logits

    def densify_and_prune(
        self,
        xyz_grad: torch.Tensor,
        grad_threshold: float,
        min_opacity: float,
        scene_extent: float,
        scale_threshold: float,
        max_points: int,
        max_screen_radius: float,
        split_factor: int = 2,
        scale_shrink: float = 1.6,
        clone_jitter: float = 0.35,
        world_prune_scale: float = 0.15,
    ) -> dict:
        changed = False
        stats = {
            "changed": False,
            "cloned": 0,
            "split_parents": 0,
            "split_children": 0,
            "pruned": 0,
            "final_points": self.num_points,
        }
        device = self.xyz.device

        mean_grad = self.xyz_gradient_accum / self.denom.clamp_min(1.0)
        mean_grad = torch.nan_to_num(mean_grad, nan=0.0, posinf=0.0, neginf=0.0)
        xyz_grad = xyz_grad.to(device)
        xyz_grad_norm = torch.norm(xyz_grad, dim=-1)

        opacity = self.opacity.detach().squeeze(-1)
        scales = self.scaling.detach().max(dim=-1).values
        scale_limit = scale_threshold * scene_extent
        world_prune_limit = world_prune_scale * scene_extent

        prune_mask = opacity < min_opacity
        prune_mask |= self.max_radii2D > max_screen_radius
        if world_prune_scale > 0.0:
            prune_mask |= scales > world_prune_limit

        clone_mask = mean_grad >= grad_threshold
        clone_mask &= scales <= scale_limit
        clone_mask &= opacity >= max(min_opacity * 1.5, 0.05)
        clone_mask &= ~prune_mask

        split_mask = mean_grad >= grad_threshold
        split_mask &= scales > scale_limit
        split_mask &= opacity >= max(min_opacity * 1.5, 0.05)
        split_mask &= ~prune_mask

        capacity = max(0, max_points - int((~prune_mask).sum().item()))
        if capacity <= 0 and not prune_mask.any():
            return stats

        clone_idx = torch.nonzero(clone_mask, as_tuple=False).squeeze(-1)
        split_idx = torch.nonzero(split_mask, as_tuple=False).squeeze(-1)

        clone_quota = capacity
        split_quota = capacity
        if split_idx.numel() > 0:
            max_split_seeds = max(0, capacity // max(split_factor, 1))
            if max_split_seeds < split_idx.numel():
                topk = torch.topk(mean_grad[split_idx], k=max_split_seeds, largest=True).indices
                split_idx = split_idx[topk]
            split_quota = max(0, capacity - split_idx.numel() * split_factor)

        if clone_idx.numel() > 0 and split_quota < clone_idx.numel():
            keep = max(0, split_quota)
            if keep > 0:
                topk = torch.topk(mean_grad[clone_idx], k=keep, largest=True).indices
                clone_idx = clone_idx[topk]
            else:
                clone_idx = clone_idx[:0]

        # Do not disable pruning; allow it even with few points to prevent unbounded growth.
        # The capacity guard above already prevents pruning all points to zero.
        selected_split_mask = torch.zeros_like(split_mask)
        if split_idx.numel() > 0:
            selected_split_mask[split_idx] = True

        stats["pruned"] = int(prune_mask.sum().item())
        stats["cloned"] = int(clone_idx.numel())
        stats["split_parents"] = int(split_idx.numel())
        stats["split_children"] = int(split_idx.numel() * split_factor)

        old_xyz = self.xyz.detach()
        old_features_dc = self.features_dc.detach()
        old_features_rest = self.features_rest.detach()
        old_scaling = self.scaling_logits.detach()
        old_rotation = self.rotation.detach()
        old_opacity = self.opacity_logits.detach()

        keep_base_mask = ~prune_mask & ~selected_split_mask
        new_xyz_all = [old_xyz[keep_base_mask]]
        new_features_dc_all = [old_features_dc[keep_base_mask]]
        new_features_rest_all = [old_features_rest[keep_base_mask]]
        new_scaling_all = [old_scaling[keep_base_mask]]
        new_rotation_all = [old_rotation[keep_base_mask]]
        new_opacity_all = [old_opacity[keep_base_mask]]

        if clone_idx.numel() > 0:
            direction = xyz_grad[clone_idx]
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            step = self.scaling.detach()[clone_idx].mean(dim=-1, keepdim=True) * clone_jitter
            new_xyz = old_xyz[clone_idx] + direction * step
            new_features_dc = old_features_dc[clone_idx]
            new_features_rest = old_features_rest[clone_idx]
            new_scaling = old_scaling[clone_idx]
            new_rotation = old_rotation[clone_idx]
            new_opacity = old_opacity[clone_idx]
            new_xyz_all.append(new_xyz)
            new_features_dc_all.append(new_features_dc)
            new_features_rest_all.append(new_features_rest)
            new_scaling_all.append(new_scaling)
            new_rotation_all.append(new_rotation)
            new_opacity_all.append(new_opacity)
            changed = True

        if split_idx.numel() > 0:
            split_children = self._split_points(split_idx, split_factor=split_factor, scale_shrink=scale_shrink)
            new_xyz_all.append(split_children[0])
            new_features_dc_all.append(split_children[1])
            new_features_rest_all.append(split_children[2])
            new_scaling_all.append(split_children[3])
            new_rotation_all.append(split_children[4])
            new_opacity_all.append(split_children[5])
            changed = True

        if prune_mask.any():
            changed = True

        if changed:
            old_xyz = torch.cat(new_xyz_all, dim=0)
            old_features_dc = torch.cat(new_features_dc_all, dim=0)
            old_features_rest = torch.cat(new_features_rest_all, dim=0)
            old_scaling = torch.cat(new_scaling_all, dim=0)
            old_rotation = torch.cat(new_rotation_all, dim=0)
            old_opacity = torch.cat(new_opacity_all, dim=0)
            if old_xyz.shape[0] > max_points:
                keep = max_points
                keep_indices = torch.topk(torch.sigmoid(old_opacity).squeeze(-1), k=keep, largest=True).indices
                old_xyz = old_xyz[keep_indices]
                old_features_dc = old_features_dc[keep_indices]
                old_features_rest = old_features_rest[keep_indices]
                old_scaling = old_scaling[keep_indices]
                old_rotation = old_rotation[keep_indices]
                old_opacity = old_opacity[keep_indices]
            self._set_parameters(old_xyz, old_features_dc, old_features_rest, old_scaling, old_rotation, old_opacity)
        stats["changed"] = changed
        stats["final_points"] = self.num_points
        return stats

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
