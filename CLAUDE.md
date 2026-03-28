# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A pure PyTorch implementation of 3D Gaussian Splatting for novel view synthesis. Uses tile-based rendering (64×64 tiles) with a fixed 16,384 Gaussian count. No adaptive densification/pruning — simpler than the original INRIA implementation but achieves comparable quality (~39 PSNR on synthetic data).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate torch-splatting
# simple-knn submodule must be cloned (--recursive)
```

## Running

```bash
# Train on the sample dataset (B075X65R3X.zip, extract first)
python train.py
# Checkpoints saved to result/test/model-{5000,10000,...,final}.pt

# Inference: load a checkpoint and render one view to a PNG
python infer.py --ckpt result/test/model-final.pt --data ./B075X65R3X --output output.png --camera 0
```

Training runs ~25k iterations at 512×512 on a fixed Gaussian set. Takes ~2 hours on RTX 2080Ti.

## Architecture

### Data Flow

```
info.json + images/depth/alpha
    → data_utils.read_all()
    → point_utils.get_point_clouds()   # depth → 3D point cloud (2^14 points)
    → GaussModel.create_from_pcd()     # initialize learnable Gaussian params
    → GSSTrainer loop                  # random camera → render → loss → backprop
```

### Key Modules

**`gaussian_splatting/gauss_model.py` — GaussModel**
Holds all learnable Gaussian parameters as `nn.Parameter`:
- `_xyz` [N,3]: 3D positions
- `_features_dc` [N,1,3] + `_features_rest` [N,15,3]: Spherical Harmonics (degree 4)
- `_scaling` [N,3]: log-space scale
- `_rotation` [N,4]: unit quaternions
- `_opacity` [N,1]: pre-sigmoid opacity

**`gaussian_splatting/gauss_render.py` — GaussRenderer**
Pure PyTorch rendering pipeline:
1. `projection_ndc()` — world → camera → NDC → viewport
2. `build_covariance_2d()` — EWA splatting: project 3D covariance to screen space
3. `get_radius()` — max eigenvalue of 2D covariance → splat radius
4. `render()` — tile-based rasterization: assign Gaussians to 64×64 tiles, depth-sort within each tile, alpha composite front-to-back

**`gaussian_splatting/trainer.py` — Trainer (base)**
Abstract base handling Adam optimization, optional FP16 mixed precision, and HuggingFace Accelerate for distributed training. Subclasses implement `on_train_step()` and `on_evaluate_step()`.

**`train.py` — GSSTrainer**
Concrete trainer: samples random camera view, calls renderer, computes weighted L1 + SSIM loss (+ optional depth loss), and logs visualizations every 100 steps.

### Utility Modules (`gaussian_splatting/utils/`)

| File | Purpose |
|------|---------|
| `data_utils.py` | Load scene from `info.json` + image files |
| `camera_utils.py` | Camera projection, intrinsics/extrinsics, Blender→OpenCV convention |
| `point_utils.py` | Convert depth maps to 3D point clouds; `PointCloud` class |
| `sh_utils.py` | Spherical Harmonics evaluation; `RGB2SH()` / `eval_sh()` |
| `loss_utils.py` | L1, L2, SSIM losses |
| `blender2json.py` + `blender_script.py` | Export Blender scenes to `info.json` format |

## Input Data Format

```
dataset/
├── info.json
├── image_0_rgb.png
├── image_0_depth.png
├── image_0_alpha.png
└── ...
```

`info.json` has a top-level `"images"` list; each entry has `rgb`, `pose` (4×4 c2w matrix, Blender convention), `intrinsic` (3×3), and `max_depth`. Camera poses are converted from Blender (Y-up, Z-back) to OpenCV (Y-down, Z-forward) in `camera_utils.py`.

## Notable Implementation Choices

- **Fixed Gaussian count** (2^14): no splitting/pruning/cloning as in the original paper
- **64×64 tiles** instead of 16×16 — Python loop over tiles is the bottleneck; larger tiles reduce iterations
- **Pure PyTorch KNN** (`_compute_nn_sq_distances` in `gauss_model.py`): batched `torch.cdist` in 512-point chunks replaces the original `simple_knn` CUDA extension for initialization
- Rendering is differentiable via PyTorch autograd (no custom CUDA kernels)
- `GaussModel.create_from_state_dict(state_dict)` reconstructs a model from a saved checkpoint without needing the original point cloud (used by `infer.py`)
