import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from colmap_reader import CameraInfo, read_colmap_scene_info
from gs_model import GaussianModel


class Render(nn.Module):
    def __init__(self, camera_infos: list[CameraInfo], background_color=(0.0, 0.0, 0.0)):
        super().__init__()
        self.camera_infos = camera_infos
        self.image_names = [cam.image_name for cam in camera_infos]
        self.gt_images = [torch.from_numpy(cam.image).float() for cam in camera_infos]
        self.register_buffer(
            "world_to_cam",
            torch.tensor(np.stack([cam.tf_world_cam for cam in camera_infos]), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "focal",
            torch.tensor(np.stack([cam.focal for cam in camera_infos]), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "principal_point",
            torch.tensor(np.stack([cam.principal_point for cam in camera_infos]), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "widths",
            torch.tensor([cam.width for cam in camera_infos], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "heights",
            torch.tensor([cam.height for cam in camera_infos], dtype=torch.long),
            persistent=False,
        )
        cam_centers = []
        for cam in camera_infos:
            rot = cam.tf_world_cam[:3, :3]
            trans = cam.tf_world_cam[:3, 3]
            cam_centers.append((-rot.T @ trans).astype(np.float32))
        self.register_buffer(
            "camera_centers",
            torch.tensor(np.stack(cam_centers), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )

    def get_ground_truth(self, index: int, device: torch.device | str | None = None) -> torch.Tensor:
        image = self.gt_images[index]
        if device is not None:
            image = image.to(device)
        return image

    def project(self, xyz: torch.Tensor, camera_index: int, cov_world: torch.Tensor):
        world_to_cam = self.world_to_cam[camera_index].to(xyz.device)
        rot = world_to_cam[:3, :3]
        trans = world_to_cam[:3, 3]
        focal = self.focal[camera_index].to(xyz.device)
        principal_point = self.principal_point[camera_index].to(xyz.device)

        cam_xyz = xyz @ rot.T + trans
        z = cam_xyz[:, 2].clamp_min(1e-6)

        means_2d = torch.empty((xyz.shape[0], 2), device=xyz.device, dtype=xyz.dtype)
        means_2d[:, 0] = focal[0] * cam_xyz[:, 0] / z + principal_point[0]
        means_2d[:, 1] = focal[1] * cam_xyz[:, 1] / z + principal_point[1]

        cov_cam = rot.unsqueeze(0) @ cov_world @ rot.T.unsqueeze(0)

        jacobian = torch.zeros((xyz.shape[0], 2, 3), device=xyz.device, dtype=xyz.dtype)
        jacobian[:, 0, 0] = focal[0] / z
        jacobian[:, 0, 2] = -focal[0] * cam_xyz[:, 0] / (z * z)
        jacobian[:, 1, 1] = focal[1] / z
        jacobian[:, 1, 2] = -focal[1] * cam_xyz[:, 1] / (z * z)

        cov_2d = jacobian @ cov_cam @ jacobian.transpose(-1, -2)
        blur = torch.eye(2, device=xyz.device, dtype=xyz.dtype).unsqueeze(0) * 0.3
        cov_2d = cov_2d + blur
        return cam_xyz, means_2d, cov_2d

    def _prepare_gaussians(
        self,
        camera_index: int,
        gaussian_model: GaussianModel,
        scaling_modifier: float,
        min_depth: float,
        max_screen_radius: float,
        margin_pixels: float,
    ) -> dict | None:
        device = gaussian_model.xyz.device
        width = int(self.widths[camera_index].item())
        height = int(self.heights[camera_index].item())

        xyz = gaussian_model.xyz
        cov_world = gaussian_model.get_covariance(scaling_modifier=scaling_modifier)
        cam_xyz, means_2d, cov_2d = self.project(xyz, camera_index, cov_world)

        cam_center = self.camera_centers[camera_index].to(device)
        view_dirs = cam_center.unsqueeze(0) - xyz
        view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        colors = gaussian_model.get_colors(view_dirs)
        opacity = gaussian_model.opacity.squeeze(-1)

        eigvals = torch.linalg.eigvalsh(cov_2d)
        max_eig = eigvals[:, 1].clamp_min(1e-8)
        radius = (3.0 * torch.sqrt(max_eig)).clamp(max=max_screen_radius)

        valid = cam_xyz[:, 2] > min_depth
        valid &= opacity > 1e-4
        valid &= radius > 0.25
        valid &= means_2d[:, 0] + radius >= -margin_pixels
        valid &= means_2d[:, 0] - radius < width + margin_pixels
        valid &= means_2d[:, 1] + radius >= -margin_pixels
        valid &= means_2d[:, 1] - radius < height + margin_pixels

        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return None

        order = torch.argsort(cam_xyz[valid_indices, 2], descending=False)
        valid_indices = valid_indices[order]
        return {
            "indices": valid_indices,
            "cam_xyz": cam_xyz,
            "means_2d": means_2d,
            "cov_2d": cov_2d,
            "colors": colors,
            "opacity": opacity,
            "radius": radius,
            "width": width,
            "height": height,
        }

    def render_camera(
        self,
        camera_index: int,
        gaussian_model: GaussianModel,
        scaling_modifier: float = 1.0,
        min_depth: float = 0.1,
        alpha_threshold: float = 1.0 / 255.0,
        max_screen_radius: float = 64.0,
        margin_pixels: float = 1.0,
        tile_size: int = 64,
    ) -> dict:
        device = gaussian_model.xyz.device
        background = self.background_color.to(device)

        prepared = self._prepare_gaussians(
            camera_index=camera_index,
            gaussian_model=gaussian_model,
            scaling_modifier=scaling_modifier,
            min_depth=min_depth,
            max_screen_radius=max_screen_radius,
            margin_pixels=margin_pixels,
        )
        width = int(self.widths[camera_index].item())
        height = int(self.heights[camera_index].item())
        if prepared is None:
            rgb = torch.ones((height, width, 3), device=device) * background
            alpha = torch.zeros((height, width), device=device)
            depth = torch.zeros((height, width), device=device)
            return {"rgb": rgb, "alpha": alpha, "depth": depth}

        valid_indices = prepared["indices"]
        cam_xyz = prepared["cam_xyz"]
        means_2d = prepared["means_2d"]
        cov_2d = prepared["cov_2d"]
        colors = prepared["colors"]
        opacity = prepared["opacity"]
        radius = prepared["radius"]
        radii = torch.ceil(radius[valid_indices])
        means_valid = means_2d[valid_indices]
        rect_min = torch.floor(means_valid - radii[:, None])
        rect_max = torch.ceil(means_valid + radii[:, None])
        rect_min[:, 0] = rect_min[:, 0].clamp(0, width - 1)
        rect_min[:, 1] = rect_min[:, 1].clamp(0, height - 1)
        rect_max[:, 0] = rect_max[:, 0].clamp(0, width - 1)
        rect_max[:, 1] = rect_max[:, 1].clamp(0, height - 1)

        render_color = torch.ones((height, width, 3), device=device, dtype=means_2d.dtype) * background.view(1, 1, 3)
        render_depth = torch.zeros((height, width, 1), device=device, dtype=means_2d.dtype)
        render_alpha = torch.zeros((height, width, 1), device=device, dtype=means_2d.dtype)

        for h in range(0, height, tile_size):
            tile_h = min(tile_size, height - h)
            for w in range(0, width, tile_size):
                tile_w = min(tile_size, width - w)
                overlap = rect_max[:, 0] >= w
                overlap &= rect_min[:, 0] < w + tile_w
                overlap &= rect_max[:, 1] >= h
                overlap &= rect_min[:, 1] < h + tile_h
                if not torch.any(overlap):
                    continue

                tile_indices = valid_indices[overlap]
                tile_depths = cam_xyz[tile_indices, 2]
                order = torch.argsort(tile_depths, descending=False)
                tile_indices = tile_indices[order]
                tile_depths = tile_depths[order]
                tile_means = means_2d[tile_indices]
                tile_cov = cov_2d[tile_indices]
                tile_opacity = opacity[tile_indices].unsqueeze(0)
                tile_color = colors[tile_indices]

                pixel_x = torch.arange(w, w + tile_w, device=device)
                pixel_y = torch.arange(h, h + tile_h, device=device)
                yy_idx, xx_idx = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
                tile_coord = torch.stack(
                    [xx_idx.to(means_2d.dtype) + 0.5, yy_idx.to(means_2d.dtype) + 0.5],
                    dim=-1,
                ).reshape(-1, 2)

                conic = torch.linalg.inv(tile_cov)
                dx = tile_coord[:, None, :] - tile_means[None, :, :]
                mahalanobis = (
                    dx[:, :, 0] * dx[:, :, 0] * conic[:, 0, 0]
                    + dx[:, :, 1] * dx[:, :, 1] * conic[:, 1, 1]
                    + dx[:, :, 0] * dx[:, :, 1] * (conic[:, 0, 1] + conic[:, 1, 0])
                )

                support_mask = mahalanobis <= 9.0
                gauss_weight = torch.exp(-0.5 * mahalanobis) * support_mask
                alpha = (gauss_weight[..., None] * tile_opacity[..., None]).clamp(max=0.99)
                alpha = torch.where(alpha >= alpha_threshold, alpha, torch.zeros_like(alpha))
                transmittance = torch.cat(
                    [torch.ones_like(alpha[:, :1]), 1.0 - alpha[:, :-1]],
                    dim=1,
                ).cumprod(dim=1)
                weights = transmittance * alpha
                acc_alpha = weights.sum(dim=1)

                tile_rgb = (weights * tile_color.unsqueeze(0)).sum(dim=1)
                tile_rgb = tile_rgb + (1.0 - acc_alpha) * background.view(1, 3)
                tile_depth = (weights[..., 0] * tile_depths.unsqueeze(0)).sum(dim=1, keepdim=True)
                tile_depth = tile_depth / acc_alpha.clamp_min(1e-8)
                tile_depth = torch.where(acc_alpha > 0, tile_depth, torch.zeros_like(tile_depth))

                render_color[h : h + tile_h, w : w + tile_w] = tile_rgb.view(tile_h, tile_w, 3)
                render_depth[h : h + tile_h, w : w + tile_w] = tile_depth.view(tile_h, tile_w, 1)
                render_alpha[h : h + tile_h, w : w + tile_w] = acc_alpha.view(tile_h, tile_w, 1)

        return {
            "rgb": render_color.clamp(0.0, 1.0),
            "alpha": render_alpha[..., 0].clamp(0.0, 1.0),
            "depth": render_depth[..., 0],
        }

    def forward(self, indexes: torch.LongTensor, gaussian_model: GaussianModel, **kwargs) -> torch.Tensor:
        renders = []
        for index in indexes.tolist():
            renders.append(self.render_camera(index, gaussian_model, **kwargs)["rgb"])
        return torch.stack(renders, dim=0)


def split_indices(num_cameras: int, holdout_every: int) -> tuple[list[int], list[int]]:
    if holdout_every <= 0:
        return list(range(num_cameras)), []
    train_ids = [idx for idx in range(num_cameras) if idx % holdout_every != 0]
    val_ids = [idx for idx in range(num_cameras) if idx % holdout_every == 0]
    if not train_ids:
        train_ids = list(range(num_cameras))
        val_ids = []
    return train_ids, val_ids


def to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().clamp(0.0, 1.0).numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def depth_to_rgb(depth: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    depth = depth.detach().cpu()
    alpha = alpha.detach().cpu() > 0.0
    if alpha.any():
        valid_depth = depth[alpha]
        depth_min = float(valid_depth.min())
        depth_max = float(valid_depth.max())
        norm = (depth - depth_min) / max(depth_max - depth_min, 1e-6)
    else:
        norm = depth * 0.0
    depth_rgb = torch.stack([norm, norm, norm], dim=-1)
    return to_uint8(depth_rgb)


def save_panel(path: str | os.PathLike, pred: torch.Tensor, gt: torch.Tensor | None = None, alpha: torch.Tensor | None = None, depth: torch.Tensor | None = None) -> None:
    tiles = [to_uint8(pred)]
    if gt is not None:
        tiles.append(to_uint8(gt))
    if alpha is not None:
        alpha_rgb = alpha.unsqueeze(-1).repeat(1, 1, 3)
        tiles.append(to_uint8(alpha_rgb))
    if depth is not None:
        alpha_for_depth = alpha if alpha is not None else torch.ones_like(depth)
        tiles.append(depth_to_rgb(depth, alpha_for_depth))
    panel = np.concatenate(tiles, axis=1)
    Image.fromarray(panel).save(path)


def load_checkpoint(checkpoint_path: str, data_path: str | None = None, image_scale: int | None = None, max_points: int | None = None, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = checkpoint.get("config", {})
    if data_path is None:
        data_path = cfg.get("data", "data/playroom")
    if image_scale is None:
        image_scale = int(cfg.get("image_scale", 1))
    if max_points is None:
        max_points = cfg.get("max_points")
    scene = read_colmap_scene_info(data_path, image_scale=image_scale)
    model = GaussianModel(
        scene,
        sh_degree=int(checkpoint.get("model", {}).get("max_sh_degree", cfg.get("sh_degree", 3))),
        max_points=max_points,
        seed=int(cfg.get("seed", 0)),
    ).to(device)
    model.restore(checkpoint["model"])
    renderer = Render(scene.cameras, background_color=tuple(cfg.get("background_color", [0.0, 0.0, 0.0])))
    return checkpoint, scene, model, renderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure Torch Gaussian splatting renderer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint produced by train.py")
    parser.add_argument("--data", type=str, default=None, help="COLMAP scene root")
    parser.add_argument("--output", type=str, default="outputs/render", help="Directory to save rendered images")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"], help="Which camera split to render")
    parser.add_argument("--limit", type=int, default=8, help="How many views to render")
    parser.add_argument("--image-scale", type=int, default=None, help="Override image downsample factor")
    parser.add_argument("--max-points", type=int, default=None, help="Override gaussian count used when loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--holdout-every", type=int, default=None, help="Override validation stride")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint, scene, model, renderer = load_checkpoint(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        image_scale=args.image_scale,
        max_points=args.max_points,
        device=args.device,
    )

    holdout_every = args.holdout_every
    if holdout_every is None:
        holdout_every = int(checkpoint.get("config", {}).get("holdout_every", 8))
    train_ids, val_ids = split_indices(len(scene.cameras), holdout_every)

    if args.split == "train":
        camera_ids = train_ids
    elif args.split == "val":
        camera_ids = val_ids if val_ids else train_ids
    else:
        camera_ids = list(range(len(scene.cameras)))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for render_idx, camera_idx in enumerate(camera_ids[: args.limit]):
            result = renderer.render_camera(camera_idx, model)
            gt = renderer.get_ground_truth(camera_idx, device=model.xyz.device)
            filename = f"{render_idx:03d}_{renderer.image_names[camera_idx]}.png"
            save_panel(output_dir / filename, result["rgb"], gt=gt, alpha=result["alpha"], depth=result["depth"])
            print(f"saved {output_dir / filename}")


if __name__ == "__main__":
    main()
