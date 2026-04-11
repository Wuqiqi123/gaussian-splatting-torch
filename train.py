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
from render import Render, save_panel, split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pure Torch Gaussian splats on a COLMAP scene")
    parser.add_argument("--data", type=str, default="data/playroom", help="COLMAP scene root")
    parser.add_argument("--output", type=str, default="outputs/playroom", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--image-scale", type=int, default=8, help="Downsample factor for training images")
    parser.add_argument("--max-points", type=int, default=6000, help="Limit initial COLMAP points for pure Torch training speed")
    parser.add_argument("--holdout-every", type=int, default=8, help="Use every Nth image for validation")
    parser.add_argument("--white-background", action="store_true", help="Composite on white instead of black")

    parser.add_argument("--position-lr-init", type=float, default=1.6e-4)
    parser.add_argument("--feature-lr", type=float, default=2.5e-3)
    parser.add_argument("--opacity-lr", type=float, default=5.0e-2)
    parser.add_argument("--scaling-lr", type=float, default=5.0e-3)
    parser.add_argument("--rotation-lr", type=float, default=1.0e-3)

    parser.add_argument("--l1-weight", type=float, default=0.8)
    parser.add_argument("--dssim-weight", type=float, default=0.2)
    parser.add_argument("--mse-weight", type=float, default=0.2)
    parser.add_argument("--opacity-reg", type=float, default=1.0e-4)
    parser.add_argument("--scale-reg", type=float, default=1.0e-4)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--sh-upgrade-every", type=int, default=500)

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--preview-index", type=int, default=0, help="Validation view index inside split list")
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


def evaluate(renderer: Render, model: GaussianModel, camera_ids: list[int], max_views: int = 4) -> dict:
    if not camera_ids:
        return {"l1": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0}

    model.eval()
    stats = {"l1": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    count = 0
    with torch.no_grad():
        for camera_idx in camera_ids[:max_views]:
            result = renderer.render_camera(camera_idx, model)
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


def save_checkpoint(path: Path, model: GaussianModel, args: argparse.Namespace, iteration: int) -> None:
    payload = {
        "iteration": iteration,
        "config": vars(args),
        "model": model.capture(),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output)
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    scene = read_colmap_scene_info(args.data, image_scale=args.image_scale)
    background = (1.0, 1.0, 1.0) if args.white_background else (0.0, 0.0, 0.0)

    model = GaussianModel(
        scene,
        sh_degree=args.sh_degree,
        max_points=args.max_points,
        seed=args.seed,
    ).to(args.device)
    renderer = Render(scene.cameras, background_color=background)
    optimizer = model.training_setup(args)

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
        if args.sh_upgrade_every > 0 and iteration % args.sh_upgrade_every == 0:
            model.oneupSHdegree()

        camera_idx = random.choice(train_ids)
        result = renderer.render_camera(camera_idx, model)
        gt = renderer.get_ground_truth(camera_idx, device=model.xyz.device)

        loss_l1 = F.l1_loss(result["rgb"], gt)
        loss_ssim = 1.0 - ssim(result["rgb"], gt)
        loss_mse = F.mse_loss(result["rgb"], gt)
        loss = (
            args.l1_weight * loss_l1
            + args.dssim_weight * loss_ssim
            + args.mse_weight * loss_mse
            + args.opacity_reg * model.opacity.mean()
            + args.scale_reg * model.scaling.mean()
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration % args.log_every == 0 or iteration == 1:
            psnr = mse_to_psnr(loss_mse.item())
            print(
                f"iter={iteration:05d} loss={loss.item():.6f} "
                f"l1={loss_l1.item():.6f} dssim={loss_ssim.item():.6f} mse={loss_mse.item():.6f} "
                f"psnr={psnr:.2f}dB sh={model.active_sh_degree}"
            )

        if iteration % args.preview_every == 0 or iteration == 1 or iteration == args.iterations:
            model.eval()
            with torch.no_grad():
                preview = renderer.render_camera(preview_idx, model)
                preview_gt = renderer.get_ground_truth(preview_idx, device=model.xyz.device)
                save_panel(
                    preview_dir / f"iter_{iteration:05d}.png",
                    preview["rgb"],
                    gt=preview_gt,
                    alpha=preview["alpha"],
                    depth=preview["depth"],
                )
            model.train()

            val_stats = evaluate(renderer, model, val_ids if val_ids else train_ids)
            print(
                f"eval iter={iteration:05d} val_l1={val_stats['l1']:.6f} "
                f"val_ssim={val_stats['ssim']:.6f} val_mse={val_stats['mse']:.6f} val_psnr={val_stats['psnr']:.2f}dB"
            )
            if val_stats["psnr"] > best_val_psnr:
                best_val_psnr = val_stats["psnr"]
                save_checkpoint(checkpoint_dir / "best.pt", model, args, iteration)

        if iteration % args.save_every == 0 or iteration == args.iterations:
            save_checkpoint(checkpoint_dir / f"iter_{iteration:05d}.pt", model, args, iteration)
            save_checkpoint(checkpoint_dir / "latest.pt", model, args, iteration)

    print(f"Training finished. Latest checkpoint: {checkpoint_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
