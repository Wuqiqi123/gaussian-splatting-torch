"""
Inference script: load a trained Gaussian Splatting checkpoint and render one view to a PNG.

Usage:
    python infer.py --ckpt result/test/model-final.pt \
                    --data ./B075X65R3X \
                    --output output.png \
                    --camera 0 \
                    --resize 0.5
"""

import argparse
import torch
import numpy as np
import imageio

from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render import GaussRenderer
from gaussian_splatting.utils.data_utils import read_all
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera


def render_to_image(ckpt_path, data_folder, output_path, camera_index=0, resize=0.5, device='cuda'):
    # ── load camera data ──────────────────────────────────────────────────────
    data = read_all(data_folder, resize_factor=resize)
    data = {k: v.to(device) for k, v in data.items()}
    n_cameras = len(data['camera'])
    if camera_index >= n_cameras:
        raise ValueError(f"camera_index {camera_index} out of range (dataset has {n_cameras} cameras)")

    camera = to_viewpoint_camera(data['camera'][camera_index])

    # ── load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model']
    step = ckpt.get('step', '?')
    model = GaussModel.create_from_state_dict(state_dict, device=device)
    model.eval()
    print(f"Loaded checkpoint from step {step} — {model._xyz.shape[0]} Gaussians")

    # ── render ────────────────────────────────────────────────────────────────
    renderer = GaussRenderer(active_sh_degree=model.max_sh_degree, white_bkgd=True)
    with torch.no_grad():
        out = renderer(pc=model, camera=camera)

    # ── save ──────────────────────────────────────────────────────────────────
    img = out['render'].detach().cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    imageio.imwrite(output_path, img)
    print(f"Saved rendered image → {output_path}  ({img.shape[1]}×{img.shape[0]})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',   required=True,          help='Path to .pt checkpoint')
    parser.add_argument('--data',   default='./B075X65R3X', help='Dataset folder with info.json')
    parser.add_argument('--output', default='output.png',   help='Output image path')
    parser.add_argument('--camera', type=int, default=0,    help='Camera index to render')
    parser.add_argument('--resize', type=float, default=0.5, help='Resize factor (must match training)')
    parser.add_argument('--device', default='cuda',         help='torch device')
    args = parser.parse_args()

    render_to_image(
        ckpt_path=args.ckpt,
        data_folder=args.data,
        output_path=args.output,
        camera_index=args.camera,
        resize=args.resize,
        device=args.device,
    )
