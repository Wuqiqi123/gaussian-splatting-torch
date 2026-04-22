import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from colmap_reader import CameraInfo, read_colmap_scene_info
from gs_model import GaussianModel

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except Exception as exc:  # pragma: no cover - import availability depends on local CUDA build.
    GaussianRasterizationSettings = None
    GaussianRasterizer = None
    _RASTERIZER_IMPORT_ERROR = exc
else:
    _RASTERIZER_IMPORT_ERROR = None


def require_diff_gaussian_rasterizer() -> None:
    if GaussianRasterizer is None or GaussianRasterizationSettings is None:
        raise ImportError(
            "diff-gaussian-rasterization is required. Install it with "
            "`pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git`."
        ) from _RASTERIZER_IMPORT_ERROR


def get_projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float) -> torch.Tensor:
    tan_half_fov_y = math.tan(fov_y * 0.5)
    tan_half_fov_x = math.tan(fov_x * 0.5)

    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    proj = torch.zeros((4, 4), dtype=torch.float32)
    proj[0, 0] = 2.0 * znear / (right - left)
    proj[1, 1] = 2.0 * znear / (top - bottom)
    proj[0, 2] = (right + left) / (right - left)
    proj[1, 2] = (top + bottom) / (top - bottom)
    proj[3, 2] = 1.0
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    return proj


def normalize_render_mode(render_mode: str) -> str:
    if render_mode not in {"diff", "non_tile", "tile"}:
        raise ValueError(f"Unsupported render mode '{render_mode}'")
    return render_mode


class Render(nn.Module):
    def __init__(self, camera_infos: list[CameraInfo], background_color=(0.0, 0.0, 0.0), znear: float = 0.01, zfar: float = 100.0):
        super().__init__()
        require_diff_gaussian_rasterizer()
        self.camera_infos = camera_infos
        self.image_names = [cam.image_name for cam in camera_infos]
        self.gt_images = [torch.from_numpy(cam.image).float() for cam in camera_infos]

        world_view = []
        full_proj = []
        camera_centers = []
        fovs = []
        for cam in camera_infos:
            view = torch.tensor(cam.tf_world_cam, dtype=torch.float32).transpose(0, 1)
            proj = get_projection_matrix(znear=znear, zfar=zfar, fov_x=float(cam.fov[0]), fov_y=float(cam.fov[1])).transpose(0, 1)
            world_view.append(view)
            full_proj.append(view @ proj)
            camera_centers.append(torch.linalg.inv(view)[3, :3])
            fovs.append(cam.fov.astype(np.float32))

        self.register_buffer("world_view_transform", torch.stack(world_view), persistent=False)
        self.register_buffer("full_proj_transform", torch.stack(full_proj), persistent=False)
        self.register_buffer("camera_centers", torch.stack(camera_centers), persistent=False)
        self.register_buffer("fov", torch.tensor(np.stack(fovs), dtype=torch.float32), persistent=False)
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
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        self.znear = float(znear)
        self.zfar = float(zfar)

    def get_ground_truth(self, index: int, device: torch.device | str | None = None) -> torch.Tensor:
        image = self.gt_images[index]
        if device is not None:
            image = image.to(device)
        return image

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
        bbox_pad_pixels: float = 1.5,
        render_mode: str = "diff",
    ) -> dict:
        del min_depth, alpha_threshold, max_screen_radius, margin_pixels, tile_size, bbox_pad_pixels
        normalize_render_mode(render_mode)

        device = gaussian_model.xyz.device
        if device.type != "cuda":
            raise RuntimeError("diff-gaussian-rasterization requires a CUDA device")

        means2d = torch.zeros_like(gaussian_model.xyz, dtype=gaussian_model.xyz.dtype, requires_grad=True, device=device)
        try:
            means2d.retain_grad()
        except RuntimeError:
            pass

        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.heights[camera_index].item()),
            image_width=int(self.widths[camera_index].item()),
            tanfovx=math.tan(float(self.fov[camera_index, 0].item()) * 0.5),
            tanfovy=math.tan(float(self.fov[camera_index, 1].item()) * 0.5),
            bg=self.background_color.to(device=device, dtype=gaussian_model.xyz.dtype),
            scale_modifier=scaling_modifier,
            viewmatrix=self.world_view_transform[camera_index].to(device=device, dtype=gaussian_model.xyz.dtype),
            projmatrix=self.full_proj_transform[camera_index].to(device=device, dtype=gaussian_model.xyz.dtype),
            sh_degree=int(gaussian_model.active_sh_degree),
            campos=self.camera_centers[camera_index].to(device=device, dtype=gaussian_model.xyz.dtype),
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        color, radii = rasterizer(
            means3D=gaussian_model.xyz,
            means2D=means2d,
            opacities=gaussian_model.opacity,
            shs=gaussian_model.features,
            scales=gaussian_model.scaling,
            rotations=nn.functional.normalize(gaussian_model.rotation, dim=-1),
            colors_precomp=None,
            cov3D_precomp=None,
        )

        rgb = color.permute(1, 2, 0) if color.ndim == 3 and color.shape[0] in {1, 3} else color
        visible_indices = torch.nonzero(radii > 0, as_tuple=False).squeeze(-1)

        return {
            "rgb": rgb.clamp(0.0, 1.0),
            "alpha": None,
            "depth": None,
            "viewspace_points": means2d,
            "visible_indices": visible_indices,
            "radii": radii,
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


def save_panel(
    path: str | os.PathLike,
    pred: torch.Tensor,
    gt: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    depth: torch.Tensor | None = None,
    include_alpha: bool = True,
    include_depth: bool = True,
) -> None:
    tiles = [to_uint8(pred)]
    if gt is not None:
        tiles.append(to_uint8(gt))
    if include_alpha and alpha is not None:
        alpha_rgb = alpha.unsqueeze(-1).repeat(1, 1, 3)
        tiles.append(to_uint8(alpha_rgb))
    if include_depth and depth is not None:
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
    background_color = cfg.get("background_color")
    if background_color is None:
        background_color = [1.0, 1.0, 1.0] if cfg.get("white_background", False) else [0.0, 0.0, 0.0]
    renderer = Render(scene.cameras, background_color=tuple(background_color))
    return checkpoint, scene, model, renderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian splatting renderer backed by diff-gaussian-rasterization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint produced by train.py")
    parser.add_argument("--data", type=str, default=None, help="COLMAP scene root")
    parser.add_argument("--output", type=str, default="outputs/render", help="Directory to save rendered images")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"], help="Which camera split to render")
    parser.add_argument("--limit", type=int, default=8, help="How many views to render")
    parser.add_argument("--image-scale", type=int, default=None, help="Override image downsample factor")
    parser.add_argument("--max-points", type=int, default=None, help="Override gaussian count used when loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--holdout-every", type=int, default=None, help="Override validation stride")
    parser.add_argument("--render-mode", type=str, default="diff", choices=["diff", "non_tile", "tile"])
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
            result = renderer.render_camera(camera_idx, model, render_mode=args.render_mode)
            gt = renderer.get_ground_truth(camera_idx, device=model.xyz.device)
            filename = f"{render_idx:03d}_{renderer.image_names[camera_idx]}.png"
            save_panel(output_dir / filename, result["rgb"], gt=gt, alpha=result["alpha"], depth=result["depth"])
            print(f"saved {output_dir / filename}")


if __name__ == "__main__":
    main()
