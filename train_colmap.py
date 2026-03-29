import torch
import numpy as np
import gaussian_splatting.utils as utils
import gaussian_splatting.utils.loss_utils as loss_utils
from gaussian_splatting.trainer import Trainer
from gaussian_splatting.utils.colmap_utils import read_colmap
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera
from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render import GaussRenderer


class GSSTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data')
        self.gaussRender = GaussRenderer(**kwargs.get('render_kwargs', {}))
        self.lambda_dssim = 0.2

    def on_train_step(self):
        ind = np.random.choice(len(self.data['camera']))
        camera = to_viewpoint_camera(self.data['camera'][ind])
        rgb    = self.data['rgb'][ind]

        out = self.gaussRender(pc=self.model, camera=camera)

        l1_loss   = loss_utils.l1_loss(out['render'], rgb)
        ssim_loss = 1.0 - loss_utils.ssim(out['render'], rgb)
        total_loss = (1 - self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss

        psnr = utils.img2psnr(out['render'], rgb)
        return total_loss, {'total': total_loss, 'l1': l1_loss, 'ssim': ssim_loss, 'psnr': psnr}

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        ind    = np.random.choice(len(self.data['camera']))
        camera = to_viewpoint_camera(self.data['camera'][ind])
        rgb    = self.data['rgb'][ind].detach().cpu().numpy()

        out      = self.gaussRender(pc=self.model, camera=camera)
        rgb_pd   = out['render'].detach().cpu().numpy()
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth_pd = (1 - depth_pd / (depth_pd.max() + 1e-6))
        depth_pd = plt.get_cmap('jet')(depth_pd)[..., :3]

        image = np.concatenate([rgb, rgb_pd, depth_pd], axis=1)
        utils.imwrite(str(self.results_folder / f'image-{self.step}.png'), image)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='./dataset/playroom', help='COLMAP dataset folder')
    parser.add_argument('--result',  default='result/playroom',    help='Output folder')
    parser.add_argument('--resize',  type=float, default=0.25,     help='Resize factor (0.25 recommended for full-res DSLR)')
    parser.add_argument('--steps',   type=int,   default=30000)
    parser.add_argument('--n_gauss', type=int,   default=2**14,    help='Number of Gaussians')
    parser.add_argument('--device',  default='cuda')
    args = parser.parse_args()

    # ── data ──────────────────────────────────────────────────────────────────
    data = read_colmap(args.data, resize_factor=args.resize)
    data['rgb']    = data['rgb'].to(args.device)
    data['camera'] = data['camera'].to(args.device)

    # ── model ─────────────────────────────────────────────────────────────────
    points = data.pop('points').random_sample(args.n_gauss)
    print(f"Initialising {args.n_gauss} Gaussians from {len(points.coords)} sparse points")

    model = GaussModel(debug=False)
    model.create_from_pcd(points, device=args.device)

    # ── train ─────────────────────────────────────────────────────────────────
    trainer = GSSTrainer(
        model=model,
        data=data,
        train_batch_size=1,
        train_num_steps=args.steps,
        i_image=200,
        i_save=5000,
        train_lr=1e-3,
        amp=False,
        fp16=False,
        results_folder=args.result,
        render_kwargs={'white_bkgd': True},
    )

    trainer.on_evaluate_step()   # render once before training to sanity-check
    trainer.train()
