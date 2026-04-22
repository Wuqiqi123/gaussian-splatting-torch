# gaussian-splatting-torch

A clean PyTorch re-implementation of [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) using the official `diff-gaussian-rasterization` CUDA backend.

## Features

- COLMAP scene loading with automatic train/val split
- Adaptive density control: clone, split, prune
- Exponential LR decay for xyz (matches official schedule)
- Spherical harmonics up to degree 3, upgraded progressively during training
- Periodic opacity reset
- Preview images and checkpoints saved during training
- Validation PSNR/SSIM tracked; best checkpoint saved automatically

## Requirements

```
torch >= 2.0
diff-gaussian-rasterization  # submodule, build with pip install ./diff-gaussian-rasterization
scipy
Pillow
```

Install the rasterizer:

```bash
pip install ./diff-gaussian-rasterization
```

## Data

Expects a COLMAP scene with the standard layout:

```
data/playroom/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
  images/
    *.jpg / *.png
```

## Training

```bash
bash run_train.sh
```

Or manually:

```bash
python train.py \
  --data data/playroom \
  --output outputs/playroom \
  --iterations 30000 \
  --sh-degree 3 \
  --max-gaussians 1000000
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--iterations` | 10000 | Total training steps |
| `--sh-degree` | 3 | Max spherical harmonics degree |
| `--max-gaussians` | 1000000 | Gaussian count cap |
| `--densify-from` | 500 | Start densification at this step |
| `--densify-until` | 15000 | Stop densification at this step |
| `--opacity-reset-interval` | 3000 | Reset opacity every N steps |
| `--holdout-every` | 8 | Every Nth camera goes to val split |
| `--image-scale` | 1 | Downsample factor (1 = full res) |

## Outputs

```
outputs/playroom/
  checkpoints/
    iter_05000.pt
    iter_10000.pt
    latest.pt
    best.pt          # best val PSNR
  previews/
    iter_00500.png   # side-by-side render vs GT
    ...
  run_config.json
```

## Rendering

```bash
bash run_render.sh
```

Or manually:

```bash
python render.py \
  --checkpoint outputs/playroom/checkpoints/latest.pt \
  --data data/playroom \
  --output outputs/playroom/render_val \
  --split val
```

## Results (playroom, 30k iterations)

| Metric | Value |
|---|---|
| Train PSNR | ~35–38 dB |
| Val PSNR | ~30.4 dB |
| Val SSIM | ~0.920 |
| Gaussians | ~842k |

## File Overview

| File | Description |
|---|---|
| `train.py` | Training loop, densification, logging |
| `gs_model.py` | GaussianModel: parameters, optimizer, densify/prune |
| `render.py` | Render wrapper around diff-gaussian-rasterization |
| `colmap_reader.py` | COLMAP binary reader, camera/image parsing |
| `sh_rgb.py` | Spherical harmonics utilities |
| `run_train.sh` | Training launch script |
| `run_render.sh` | Render launch script |
