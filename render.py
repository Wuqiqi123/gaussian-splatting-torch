from torch import nn
import torch
import numpy as np
from pypose import SE3, mat2SE3
from colmap_reader import CameraInfo
from gs_model import GaussianModel
import math
from typing import List

class Render(nn.Module):
    def __init__(self, camera_infos: List[CameraInfo]):
        super(Render, self).__init__()
        self.uids = torch.stack([torch.tensor(cam_info.uid) for cam_info in camera_infos], dim=0)
        self.tf_world_cams = mat2SE3(torch.stack([torch.tensor(cam_info.tf_world_cam) for cam_info in camera_infos], dim=0).float())
        self.fovs = torch.stack([torch.tensor(cam_info.fov) for cam_info in camera_infos], dim=0).float()
        self.image_names = [cam_info.image_name for cam_info in camera_infos]
        self.gt_images = torch.stack([torch.tensor(cam_info.image) for cam_info in camera_infos], dim=0)
        self.widths = torch.tensor([cam_info.width for cam_info in camera_infos])
        self.heights = torch.tensor([cam_info.height for cam_info in camera_infos])
        self.focal = torch.stack([torch.tensor(cam_info.focal) for cam_info in camera_infos], dim=0).float()

        self.zfar = 100.0
        self.znear = 0.01

        self.tf_cams_world = self.tf_world_cams.Inv()
        self.projection_matrix = self.get_projection_matrix(znear=self.znear, zfar=self.zfar, fovs=self.fovs)
        self.full_proj_transform = self.projection_matrix @ self.tf_cams_world.matrix()

    def get_projection_matrix(self, znear, zfar, fovs):
        tan_half_fov = torch.tan(fovs / 2)

        top = tan_half_fov[:, 1] * znear
        bottom = -top
        right = tan_half_fov[:, 0] * znear
        left = -right

        P = torch.zeros((fovs.shape[0], 4, 4))

        z_sign = 1.0

        P[..., 0, 0] = 2.0 * znear / (right - left)
        P[..., 1, 1] = 2.0 * znear / (top - bottom)
        P[..., 0, 2] = (right + left) / (right - left)
        P[..., 1, 2] = (top + bottom) / (top - bottom)
        P[..., 3, 2] = z_sign
        P[..., 2, 2] = z_sign * zfar / (zfar - znear)
        P[..., 2, 3] = -(zfar * znear) / (zfar - znear)
        return P
    
    def compute_cov_2d(self, means_3d_in_camera_space, max_x_on_normalized_plane, max_y_on_normalized_plane, focal, cov_3d, tf_cams_world):
        limx = (1.3 * max_x_on_normalized_plane)[..., None, None]
        limy = (1.3 * max_y_on_normalized_plane)[..., None, None]

        x = means_3d_in_camera_space[..., 0].unsqueeze(-1)
        y = means_3d_in_camera_space[..., 1].unsqueeze(-1)
        z = means_3d_in_camera_space[..., 2].unsqueeze(-1)
        txtz = x / z
        tytz = y / z


        clamped_x = torch.clamp(txtz, min=-limx, max=limx) * z
        clamped_y = torch.clamp(tytz, min=-limy, max=limy) * z

        means_in_camera_space = torch.cat([clamped_x, clamped_y, z], dim=-1)

        # build Jacobian matrix J
        J = torch.zeros((*means_in_camera_space.shape[0:2], 3, 3), dtype=means_3d_in_camera_space.dtype, device=means_in_camera_space.device)

        focal_x = focal[..., 0][..., None]
        focal_y = focal[..., 1][..., None]
        x = means_in_camera_space[..., 0]
        y = means_in_camera_space[..., 1]
        z = means_in_camera_space[..., 2]
        J[..., 0, 0] = focal_x / z
        J[..., 0, 2] = -(focal_x * x) / (z * z)
        J[..., 1, 1] = focal_y / z
        J[..., 1, 2] = -(focal_y * y) / (z * z)
        # the third row of J is ignored

        # build transform matrix W
        W = tf_cams_world.matrix().unsqueeze(1)[..., :3, :3]

        T = J @ W

        cov_2d = T @ cov_3d @ T.transpose(-1, -2)

        return cov_2d[..., :2, :2]


    def torch_rasterize(self, means_2d, cov_2d, opacities, colors, image_size):
        """
        means_2d: [N, 2] pixel cntroids of the Gaussians
        cov_2d: [N, 2, 2] 2D covariances of the Gaussians
        opacities: [N, 1] opacities (alpha)
        colors: [N, 3] colors of the Gaussians
        image_size: (H, W)
        """
        H, W = image_size
        device = means_2d.device
        
        # 1. 生成像素网格 [H, W, 2]
        gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        pixel_coords = torch.stack([gx, gy], dim=-1).float() + 0.5  # [H, W, 2]
        
        # 2. 计算 2D 协方差的逆 (Precision Matrix)
        # det = ad - bc
        det = cov_2d[:, 0, 0] * cov_2d[:, 1, 1] - cov_2d[:, 0, 1] * cov_2d[:, 1, 0]
        inv_cov_2d = torch.zeros_like(cov_2d)
        inv_cov_2d[:, 0, 0] =  cov_2d[:, 1, 1] / det
        inv_cov_2d[:, 1, 1] =  cov_2d[:, 0, 0] / det
        inv_cov_2d[:, 0, 1] = -cov_2d[:, 0, 1] / det
        inv_cov_2d[:, 1, 0] = -cov_2d[:, 1, 0] / det

        # 3. 向量化计算所有像素与所有高斯的距离 (极其耗显存，仅建议小量点云调试)
        # 在实际 CUDA 实现中，这里会使用 Tile-based 渲染，只计算像素附近的高斯
        dx = pixel_coords[:, :, None, :] - means_2d[None, None, :, :] # [H, W, N, 2]
        
        # 计算指数部分: -0.5 * dx^T * Inv_Cov * dx
        # mahalanobis_dist = (dx @ inv_cov_2d) * dx
        intermediate = torch.einsum('hwnk,nkl->hwnl', dx, inv_cov_2d)
        mahalanobis_dist = torch.sum(intermediate * dx, dim=-1) # [H, W, N]
        
        # 4. 计算每个高斯在每个像素的权重 G_i(x)
        gaussian_weight = torch.exp(-0.5 * mahalanobis_dist)
        alpha = opacities.view(1, 1, -1) * gaussian_weight # [H, W, N]
        
        # 5. Alpha Blending (需要按深度排序)
        # 注意：这里假设输入的 means_2d 已经是按深度从近到远排好序的
        # T_i = product(1 - alpha_j)
        cumprod_1_minus_alpha = torch.cumprod(1.0 - alpha + 1e-7, dim=-1)
        # 前 i-1 项的透射率
        transmittance = torch.cat([torch.ones((H, W, 1), device=device), cumprod_1_minus_alpha[:, :, :-1]], dim=-1)
        
        weights = alpha * transmittance # [H, W, N]
        
        # 6. 合成最终颜色
        rendered_image = torch.einsum('hwn,nc->hwc', weights, colors)
        
        return rendered_image


    def forward(self, indexes:  torch.LongTensor, gaussian_model: GaussianModel, scaling_modifier = 1.0, min_depth = 0.01):
        '''
        Render the scene from the camera at the given index using the provided GaussianModel.
        '''
        assert indexes.dim() == 1
        batch_size = indexes.shape[0]
        proj_mats = self.full_proj_transform[indexes]
        tf_cams_world = self.tf_cams_world[indexes]
        widths = self.widths[indexes]
        heights = self.heights[indexes]
        focal = self.focal[indexes]

        homogeneous_xyz = torch.cat([gaussian_model.xyz, torch.ones((gaussian_model.xyz.shape[0], 1), device=gaussian_model.xyz.device)], dim=-1)
        means_3d_in_camera_space = tf_cams_world.unsqueeze(1) @ homogeneous_xyz.unsqueeze(0)
        with torch.no_grad():
            is_min_depth_satisfied = means_3d_in_camera_space[:, 2] >= min_depth


        cov_3d = gaussian_model.get_covariance(scaling_modifier).unsqueeze(0)

        max_x_on_normalized_plane = (0.5 * widths) / focal[..., 0]
        max_y_on_normalized_plane = (0.5 * widths) / focal[..., 1]
        cov_2d = self.compute_cov_2d(
            means_3d_in_camera_space,
            max_x_on_normalized_plane=max_x_on_normalized_plane,
            max_y_on_normalized_plane=max_y_on_normalized_plane,
            focal=focal,
            cov_3d=cov_3d,
            tf_cams_world=tf_cams_world,
        )

        self.torch_rasterize(
            means_2d = (means_3d_in_camera_space[..., :2] / means_3d_in_camera_space[..., 2:3]),
            cov_2d = cov_2d,
            opacities = gaussian_model.opacities.squeeze(-1),
            colors = gaussian_model.colors,
            image_size = (heights[0].item(), widths[0].item())
        )
        




if __name__ == "__main__":
    
    from colmap_reader import read_colmap_scene_info
    from viz_colmap import create_visual_scene
    import trimesh

    scene = read_colmap_scene_info("data/playroom")

    model = GaussianModel(scene)

    render = Render(scene.cameras)

    render(torch.tensor([0, 1, 3]), model)
