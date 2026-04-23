import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from colmap_reader import read_colmap_scene_info
from gs_model import GaussianModel
from lighting_probes import LightingProbes, build_probes_from_scene
from render import Render, save_panel, split_indices

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gaussian splats on a COLMAP scene with diff-gaussian-rasterization")
    parser.add_argument("--data", type=str, default="data/playroom", help="COLMAP scene root")
    parser.add_argument("--output", type=str, default="outputs/playroom", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--image-scale", type=int, default=1, help="Downsample factor for training images")
    parser.add_argument("--max-points", type=int, default=13000, help="Limit initial COLMAP points for training speed")
    parser.add_argument("--holdout-every", type=int, default=8, help="Use every Nth image for validation")
    parser.add_argument("--white-background", action="store_true", help="Composite on white instead of black")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="diff",
        choices=["diff", "non_tile", "tile"],
        help="Rasterizer backend. non_tile/tile are accepted as legacy aliases for the diff rasterizer.",
    )

    parser.add_argument("--position-lr-init", type=float, default=1.6e-4)
    parser.add_argument("--position-lr-final", type=float, default=1.6e-6)
    parser.add_argument("--position-lr-delay-mult", type=float, default=0.01)
    parser.add_argument("--position-lr-max-steps", type=int, default=30_000)
    parser.add_argument("--feature-lr", type=float, default=2.5e-3)
    parser.add_argument("--opacity-lr", type=float, default=2.5e-2)
    parser.add_argument("--scaling-lr", type=float, default=5.0e-3)
    parser.add_argument("--rotation-lr", type=float, default=1.0e-3)

    parser.add_argument("--l1-weight", type=float, default=0.8)
    parser.add_argument("--dssim-weight", type=float, default=0.2)
    parser.add_argument("--opacity-reg", type=float, default=1.0e-4)
    parser.add_argument("--scale-reg", type=float, default=1.0e-4)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--sh-upgrade-every", type=int, default=1000)

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--preview-index", type=int, default=0, help="Validation view index inside split list")

    parser.add_argument("--densify-from", type=int, default=500)
    parser.add_argument("--densify-until", type=int, default=15000)
    parser.add_argument("--densify-interval", type=int, default=100)
    parser.add_argument("--densify-grad-threshold", type=float, default=2e-4)
    parser.add_argument("--densify-scale-threshold", type=float, default=0.01)
    parser.add_argument("--prune-opacity-threshold", type=float, default=0.005)
    parser.add_argument("--prune-screen-radius", type=float, default=20.0)
    parser.add_argument("--world-prune-scale", type=float, default=0.1)
    parser.add_argument("--max-gaussians", type=int, default=1_000_000)
    parser.add_argument("--split-factor", type=int, default=2)
    parser.add_argument("--split-scale-factor", type=float, default=1.6)
    parser.add_argument("--clone-jitter", type=float, default=0.35)
    parser.add_argument("--opacity-reset-interval", type=int, default=3000)
    parser.add_argument("--opacity-reset-max", type=float, default=0.01)

    # Lighting probes
    parser.add_argument("--probe-grid-size", type=int, default=3, help="Probes per axis (total = N^3)")
    parser.add_argument("--probe-cubemap-res", type=int, default=8, help="Cubemap face resolution")
    parser.add_argument("--probe-lr", type=float, default=1e-3, help="Learning rate for probe cubemaps")
    parser.add_argument("--probe-k-nearest", type=int, default=4, help="Probes to blend per Gaussian")
    parser.add_argument("--no-probes", action="store_true", help="Disable lighting probes")

    # W&B
    parser.add_argument("--wandb-project", type=str, default="gaussian-splatting", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (defaults to output dir name)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mse_to_psnr(mse: float) -> float:
    mse = max(mse, 1e-10)
    return -10.0 * math.log10(mse)


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)],
        dtype=torch.float32,
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(window_2d.expand(channel, 1, window_size, window_size).contiguous())
    return window.to(device=device, dtype=dtype)


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    channel = img1.size(1)
    window = create_window(window_size, channel, img1.device, img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def evaluate(renderer: Render, model: GaussianModel, camera_ids: list[int], max_views: int = 4, render_mode: str = "non_tile") -> dict:
    if not camera_ids:
        return {"l1": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0}

    model.eval()
    stats = {"l1": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    count = 0
    with torch.no_grad():
        for camera_idx in camera_ids[:max_views]:
            result = renderer.render_camera(camera_idx, model, render_mode=render_mode)
            gt = renderer.get_ground_truth(camera_idx, device=model.xyz.device)
            l1 = F.l1_loss(result["rgb"], gt).item()
            mse = F.mse_loss(result["rgb"], gt).item()
            ssim_val = ssim(result["rgb"], gt).item()
            stats["l1"] += l1
            stats["mse"] += mse
            stats["psnr"] += mse_to_psnr(mse)
            stats["ssim"] += ssim_val
            count += 1
    model.train()
    for key in stats:
        stats[key] /= max(count, 1)
    return stats


def save_checkpoint(path: Path, model: GaussianModel, args: argparse.Namespace, iteration: int, probes=None) -> None:
    payload = {
        "iteration": iteration,
        "config": vars(args),
        "model": model.capture(),
        "probes": probes.capture() if probes is not None else None,
    }
    torch.save(payload, path)


def rebuild_optimizer(model: GaussianModel, args: argparse.Namespace, probes=None) -> tuple:
    return model.training_setup(args, probes=probes)


def update_xyz_lr(optimizer: torch.optim.Optimizer, xyz_scheduler, iteration: int) -> None:
    """Apply exponential LR decay to xyz parameter group."""
    for group in optimizer.param_groups:
        if group["name"] == "xyz":
            group["lr"] = xyz_scheduler(iteration)
            break


def _get_lr(optimizer: torch.optim.Optimizer, name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return group["lr"]
    return 0.0


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output)
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    # W&B init
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run_name or Path(args.output).name
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            dir=str(output_dir),
        )
    elif not args.no_wandb and not _WANDB_AVAILABLE:
        print("wandb not installed — run `pip install wandb` to enable logging. Continuing without it.")

    scene = read_colmap_scene_info(args.data, image_scale=args.image_scale)
    background = (1.0, 1.0, 1.0) if args.white_background else (0.0, 0.0, 0.0)

    model = GaussianModel(
        scene,
        sh_degree=args.sh_degree,
        max_points=args.max_points,
        seed=args.seed,
    ).to(args.device)

    # Lighting probes
    probes = None
    if not args.no_probes:
        points = torch.as_tensor(scene.point_cloud.points, dtype=torch.float32)
        probes = build_probes_from_scene(
            points,
            grid_size=args.probe_grid_size,
            cubemap_res=args.probe_cubemap_res,
            k_nearest=args.probe_k_nearest,
        ).to(args.device)
        print(f"Lighting probes: {probes.num_probes} probes ({args.probe_grid_size}^3), cubemap {args.probe_cubemap_res}x{args.probe_cubemap_res}")

    renderer = Render(scene.cameras, background_color=background, probes=probes)
    optimizer, xyz_scheduler = rebuild_optimizer(model, args, probes=probes)
    scene_extent = float(scene.nerf_normalization["radius"])

    train_ids, val_ids = split_indices(len(scene.cameras), args.holdout_every)
    preview_pool = val_ids if val_ids else train_ids
    preview_idx = preview_pool[min(args.preview_index, len(preview_pool) - 1)]

    metadata = {
        "num_cameras": len(scene.cameras),
        "train_views": len(train_ids),
        "val_views": len(val_ids),
        "num_points": model.num_points,
        "image_scale": args.image_scale,
        "device": args.device,
    }
    (output_dir / "run_config.json").write_text(json.dumps({**vars(args), **metadata}, indent=2), encoding="utf-8")

    print(f"Loaded {len(scene.cameras)} cameras, {model.num_points} gaussians")
    print(f"Train views: {len(train_ids)}, Val views: {len(val_ids)}")

    best_val_psnr = float("-inf")
    for iteration in range(1, args.iterations + 1):
        density_log = None
        opacity_reset_done = False

        # Update xyz exponential LR decay every iteration (matches official)
        update_xyz_lr(optimizer, xyz_scheduler, iteration)

        if args.sh_upgrade_every > 0 and iteration % args.sh_upgrade_every == 0:
            model.oneupSHdegree()

        camera_idx = random.choice(train_ids)
        result = renderer.render_camera(camera_idx, model, render_mode=args.render_mode)
        gt = renderer.get_ground_truth(camera_idx, device=model.xyz.device)

        loss_l1 = F.l1_loss(result["rgb"], gt)
        loss_ssim = 1.0 - ssim(result["rgb"], gt)
        loss = (
            args.l1_weight * loss_l1
            + args.dssim_weight * loss_ssim
            + args.opacity_reg * model.opacity.mean()
            + args.scale_reg * model.scaling.mean()
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        viewspace_points = result.get("viewspace_points")
        if viewspace_points is not None and viewspace_points.grad is not None:
            model.add_densification_stats(
                viewspace_points_grad=viewspace_points.grad.detach(),
                visible_indices=result["visible_indices"],
                radii=result["radii"].detach(),
            )
        optimizer.step()

        structure_changed = False
        if (
            args.densify_interval > 0
            and args.densify_from <= iteration <= args.densify_until
            and iteration % args.densify_interval == 0
        ):
            density_log = model.densify_and_prune(
                xyz_grad=model.xyz.grad.detach().clone() if model.xyz.grad is not None else torch.zeros_like(model.xyz),
                grad_threshold=args.densify_grad_threshold,
                min_opacity=args.prune_opacity_threshold,
                scene_extent=scene_extent,
                scale_threshold=args.densify_scale_threshold,
                max_points=args.max_gaussians,
                max_screen_radius=args.prune_screen_radius,
                split_factor=args.split_factor,
                scale_shrink=args.split_scale_factor,
                clone_jitter=args.clone_jitter,
                world_prune_scale=args.world_prune_scale,
            )
            structure_changed = density_log["changed"]
            if structure_changed:
                optimizer, xyz_scheduler = rebuild_optimizer(model, args, probes=probes)

        if (
            args.opacity_reset_interval > 0
            and iteration % args.opacity_reset_interval == 0
            and iteration <= args.densify_until
        ):
            model.reset_opacity(args.opacity_reset_max)
            optimizer, xyz_scheduler = rebuild_optimizer(model, args, probes=probes)
            structure_changed = True
            opacity_reset_done = True

        if iteration % args.log_every == 0 or iteration == 1:
            mse_val = F.mse_loss(result["rgb"].detach(), gt.detach()).item()
            psnr = mse_to_psnr(mse_val)
            log_line = (
                f"iter={iteration:05d} loss={loss.item():.6f} "
                f"l1={loss_l1.item():.6f} dssim={loss_ssim.item():.6f} mse={mse_val:.6f} "
                f"psnr={psnr:.2f}dB sh={model.active_sh_degree} points={model.num_points}"
            )
            if density_log is not None:
                log_line += (
                    f" densify[c={density_log['cloned']},"
                    f"s={density_log['split_parents']}->{density_log['split_children']},"
                    f"p={density_log['pruned']}]"
                )
            if opacity_reset_done:
                log_line += " opacity_reset=1"
            print(log_line)

            if use_wandb:
                wb_log: dict = {
                    "train/loss": loss.item(),
                    "train/loss_l1": loss_l1.item(),
                    "train/loss_dssim": loss_ssim.item(),
                    "train/psnr": psnr,
                    "train/mse": mse_val,
                    "scene/num_gaussians": model.num_points,
                    "scene/sh_degree": model.active_sh_degree,
                    "lr/xyz": _get_lr(optimizer, "xyz"),
                    "lr/f_dc": _get_lr(optimizer, "f_dc"),
                    "lr/f_rest": _get_lr(optimizer, "f_rest"),
                    "lr/opacity": _get_lr(optimizer, "opacity"),
                    "lr/scaling": _get_lr(optimizer, "scaling"),
                    "lr/rotation": _get_lr(optimizer, "rotation"),
                }
                if probes is not None:
                    wb_log["lr/probes"] = _get_lr(optimizer, "probes")
                if density_log is not None:
                    wb_log.update({
                        "densify/cloned": density_log["cloned"],
                        "densify/split_parents": density_log["split_parents"],
                        "densify/split_children": density_log["split_children"],
                        "densify/pruned": density_log["pruned"],
                        "densify/final_points": density_log["final_points"],
                    })
                if opacity_reset_done:
                    wb_log["scene/opacity_reset"] = 1
                wandb.log(wb_log, step=iteration)

        if iteration % args.preview_every == 0 or iteration == 1 or iteration == args.iterations:
            model.eval()
            preview_path = preview_dir / f"iter_{iteration:05d}.png"
            with torch.no_grad():
                preview = renderer.render_camera(preview_idx, model, render_mode=args.render_mode)
                preview_gt = renderer.get_ground_truth(preview_idx, device=model.xyz.device)
                save_panel(
                    preview_path,
                    preview["rgb"],
                    gt=preview_gt,
                    include_alpha=False,
                    include_depth=False,
                )
            model.train()

            val_stats = evaluate(renderer, model, val_ids if val_ids else train_ids, render_mode=args.render_mode)
            print(
                f"eval iter={iteration:05d} val_l1={val_stats['l1']:.6f} "
                f"val_ssim={val_stats['ssim']:.6f} val_mse={val_stats['mse']:.6f} val_psnr={val_stats['psnr']:.2f}dB"
            )
            if val_stats["psnr"] > best_val_psnr:
                best_val_psnr = val_stats["psnr"]
                save_checkpoint(checkpoint_dir / "best.pt", model, args, iteration, probes=probes)

            if use_wandb:
                wandb.log(
                    {
                        "val/psnr": val_stats["psnr"],
                        "val/ssim": val_stats["ssim"],
                        "val/l1": val_stats["l1"],
                        "val/mse": val_stats["mse"],
                        "val/best_psnr": best_val_psnr,
                        "media/preview": wandb.Image(str(preview_path), caption=f"iter {iteration}"),
                    },
                    step=iteration,
                )

        if iteration % args.save_every == 0 or iteration == args.iterations:
            save_checkpoint(checkpoint_dir / f"iter_{iteration:05d}.pt", model, args, iteration, probes=probes)
            save_checkpoint(checkpoint_dir / "latest.pt", model, args, iteration, probes=probes)

    if use_wandb:
        wandb.finish()
    print(f"Training finished. Latest checkpoint: {checkpoint_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
