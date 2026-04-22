import math
from dataclasses import dataclass, field

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


def get_expon_lr_func(lr_init, lr_final, lr_delay_mult=0.01, max_steps=30_000):
    """Exponential learning rate decay, matching the official 3DGS schedule."""
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if max_steps == 0:
            return lr_final
        t = step / max_steps
        log_lerp = math.exp((1 - t) * math.log(lr_init + 1e-20) + t * math.log(lr_final + 1e-20))
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * math.sin(0.5 * math.pi * min(t / (lr_delay_mult + 1e-8), 1.0))
        return delay_rate * log_lerp
    return helper


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
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 2.5e-3
    opacity_lr: float = 2.5e-2
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

        opacity = inverse_sigmoid(torch.full((num_points, 1), 0.1, dtype=torch.float32))

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

    def training_setup(self, training_args) -> tuple[torch.optim.Optimizer, object]:
        if isinstance(training_args, dict):
            cfg = OptimConfig(**training_args)
        else:
            cfg = OptimConfig(
                position_lr_init=getattr(training_args, "position_lr_init", OptimConfig.position_lr_init),
                position_lr_final=getattr(training_args, "position_lr_final", OptimConfig.position_lr_final),
                position_lr_delay_mult=getattr(training_args, "position_lr_delay_mult", OptimConfig.position_lr_delay_mult),
                position_lr_max_steps=getattr(training_args, "position_lr_max_steps", OptimConfig.position_lr_max_steps),
                feature_lr=getattr(training_args, "feature_lr", OptimConfig.feature_lr),
                opacity_lr=getattr(training_args, "opacity_lr", OptimConfig.opacity_lr),
                scaling_lr=getattr(training_args, "scaling_lr", OptimConfig.scaling_lr),
                rotation_lr=getattr(training_args, "rotation_lr", OptimConfig.rotation_lr),
            )

        param_groups = [
            {"params": [self.xyz], "lr": cfg.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self.features_dc], "lr": cfg.feature_lr, "name": "f_dc"},
            # Official uses feature_lr / 20 for rest SH coefficients
            {"params": [self.features_rest], "lr": cfg.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self.opacity_logits], "lr": cfg.opacity_lr, "name": "opacity"},
            {"params": [self.scaling_logits], "lr": cfg.scaling_lr, "name": "scaling"},
            {"params": [self.rotation], "lr": cfg.rotation_lr, "name": "rotation"},
        ]
        optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        # Exponential position LR scheduler matching official 3DGS
        xyz_scheduler = get_expon_lr_func(
            lr_init=cfg.position_lr_init * self.spatial_lr_scale,
            lr_final=cfg.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=cfg.position_lr_delay_mult,
            max_steps=cfg.position_lr_max_steps,
        )
        return optimizer, xyz_scheduler

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
        world_prune_scale: float = 0.1,
    ) -> dict:
        """
        Densify-and-prune closely following the official 3DGS logic:
          - Clone:  small Gaussians (max_scale <= percent_dense * extent) with high grad → copy in-place (no scale change, no jitter in position, keep opacity)
          - Split:  large Gaussians (max_scale >  percent_dense * extent) with high grad → replace with N children, scale / (0.8*N)
          - Prune:  low opacity OR large screen radius OR large world size (> 0.1 * extent)
        """
        stats = {
            "changed": False,
            "cloned": 0,
            "split_parents": 0,
            "split_children": 0,
            "pruned": 0,
            "final_points": self.num_points,
        }
        device = self.xyz.device

        # --- Compute average 2D gradient norm (matches official accumulation) ---
        grads = self.xyz_gradient_accum / self.denom.clamp_min(1.0)
        grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)

        opacity = self.opacity.detach().squeeze(-1)
        scales = self.scaling.detach().max(dim=-1).values
        scale_limit = scale_threshold * scene_extent  # percent_dense * extent

        # --- Clone: small + high grad (copy as-is, same xyz/scale/opacity) ---
        clone_mask = (grads >= grad_threshold) & (scales <= scale_limit)
        clone_idx = torch.nonzero(clone_mask, as_tuple=False).squeeze(-1)

        # --- Split: large + high grad ---
        split_mask = (grads >= grad_threshold) & (scales > scale_limit)
        split_idx = torch.nonzero(split_mask, as_tuple=False).squeeze(-1)

        # Collect new points
        new_xyz_list: list[torch.Tensor] = []
        new_fdc_list: list[torch.Tensor] = []
        new_frest_list: list[torch.Tensor] = []
        new_scale_list: list[torch.Tensor] = []
        new_rot_list: list[torch.Tensor] = []
        new_opa_list: list[torch.Tensor] = []

        # Clone: duplicate points at same position (official: new_xyz = _xyz[mask])
        if clone_idx.numel() > 0:
            new_xyz_list.append(self.xyz.detach()[clone_idx])
            new_fdc_list.append(self.features_dc.detach()[clone_idx])
            new_frest_list.append(self.features_rest.detach()[clone_idx])
            new_scale_list.append(self.scaling_logits.detach()[clone_idx])
            new_rot_list.append(self.rotation.detach()[clone_idx])
            new_opa_list.append(self.opacity_logits.detach()[clone_idx])
            stats["cloned"] = clone_idx.numel()

        # Split: sample N children around parent, scale / (0.8*N), remove parent
        if split_idx.numel() > 0:
            stds = self.scaling.detach()[split_idx].repeat(split_factor, 1)
            means = torch.zeros_like(stds)
            samples = torch.normal(mean=means, std=stds)
            rots_mat = quaternion_to_matrix(self.rotation.detach()[split_idx]).repeat(split_factor, 1, 1)
            new_xyz = torch.bmm(rots_mat, samples.unsqueeze(-1)).squeeze(-1) + self.xyz.detach()[split_idx].repeat(split_factor, 1)
            new_scale = self.scaling_inverse_activation(
                self.scaling.detach()[split_idx].repeat(split_factor, 1) / (0.8 * split_factor)
            )
            new_xyz_list.append(new_xyz)
            new_fdc_list.append(self.features_dc.detach()[split_idx].repeat(split_factor, 1, 1))
            new_frest_list.append(self.features_rest.detach()[split_idx].repeat(split_factor, 1, 1))
            new_scale_list.append(new_scale)
            new_rot_list.append(self.rotation.detach()[split_idx].repeat(split_factor, 1))
            new_opa_list.append(self.opacity_logits.detach()[split_idx].repeat(split_factor, 1))
            stats["split_parents"] = split_idx.numel()
            stats["split_children"] = split_idx.numel() * split_factor

        # --- Append new points ---
        if new_xyz_list:
            new_xyz = torch.cat(new_xyz_list, dim=0)
            new_fdc = torch.cat(new_fdc_list, dim=0)
            new_frest = torch.cat(new_frest_list, dim=0)
            new_scale = torch.cat(new_scale_list, dim=0)
            new_rot = torch.cat(new_rot_list, dim=0)
            new_opa = torch.cat(new_opa_list, dim=0)
            self.append_points(new_xyz, new_fdc, new_frest, new_scale, new_rot, new_opa)

        # --- Prune: low opacity, huge screen radius, huge world scale + split parents ---
        opacity_after = self.opacity.detach().squeeze(-1)
        scales_after = self.scaling.detach().max(dim=-1).values
        prune_mask = opacity_after < min_opacity
        if max_screen_radius is not None and max_screen_radius > 0:
            # max_radii2D only has shape of the old points; pad with zeros for new ones
            pad = self.num_points - self.max_radii2D.shape[0]
            if pad > 0:
                padded_radii = torch.cat([self.max_radii2D, torch.zeros(pad, device=device)])
            else:
                padded_radii = self.max_radii2D
            prune_mask |= padded_radii > max_screen_radius
        if world_prune_scale > 0.0:
            prune_mask |= scales_after > world_prune_scale * scene_extent

        # Remove original split parents (they were replaced by children)
        if split_idx.numel() > 0:
            # split_idx refers to the OLD point ordering before appending new points
            old_n = self.num_points - (stats["cloned"] + stats["split_children"])
            split_parent_mask = torch.zeros(self.num_points, dtype=torch.bool, device=device)
            # The old points are at indices [0, old_n)
            if old_n > 0 and split_idx.max() < old_n:
                split_parent_mask[split_idx] = True
            prune_mask = prune_mask | split_parent_mask

        stats["pruned"] = int(prune_mask.sum().item())
        changed = (stats["cloned"] + stats["split_parents"] + stats["pruned"]) > 0
        if changed:
            keep_mask = ~prune_mask
            self._set_parameters(
                self.xyz.detach()[keep_mask],
                self.features_dc.detach()[keep_mask],
                self.features_rest.detach()[keep_mask],
                self.scaling_logits.detach()[keep_mask],
                self.rotation.detach()[keep_mask],
                self.opacity_logits.detach()[keep_mask],
            )

        stats["changed"] = changed
        stats["final_points"] = self.num_points
        return stats

    @staticmethod
    def scaling_inverse_activation(x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_min(1e-8))

    def capture(self) -> dict:
        return {
            "state_dict": self.state_dict(),
            "active_sh_degree": self.active_sh_degree,
            "max_sh_degree": self.max_sh_degree,
            "num_points": self.num_points,
        }

    def restore(self, payload: dict) -> None:
        state = payload["state_dict"]
        # Resize parameters to match checkpoint shape before loading
        num_points = state["xyz"].shape[0]
        sh_dim = (self.max_sh_degree + 1) ** 2
        device = self.xyz.device
        self.xyz = nn.Parameter(torch.zeros((num_points, 3), device=device))
        self.features_dc = nn.Parameter(torch.zeros((num_points, 1, 3), device=device))
        self.features_rest = nn.Parameter(torch.zeros((num_points, sh_dim - 1, 3), device=device))
        self.scaling_logits = nn.Parameter(torch.zeros((num_points, 3), device=device))
        self.rotation = nn.Parameter(torch.zeros((num_points, 4), device=device))
        self.opacity_logits = nn.Parameter(torch.zeros((num_points, 1), device=device))
        self._reset_density_state()
        self.load_state_dict(state)
        self.active_sh_degree = int(payload.get("active_sh_degree", 0))
        self.max_sh_degree = int(payload.get("max_sh_degree", self.max_sh_degree))


if __name__ == "__main__":
    from colmap_reader import read_colmap_scene_info

    scene = read_colmap_scene_info("data/playroom", image_scale=8)
    model = GaussianModel(scene, max_points=2048)
    print("Number of Gaussians:", model.num_points)
    print("Covariance shape:", model.get_covariance().shape)
