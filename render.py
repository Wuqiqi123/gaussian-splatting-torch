from torch import nn
import torch
import numpy as np
from pypose import SE3, mat2SE3
from colmap_reader import CameraInfo
from gs_model import GaussianModel
import math

class Render(nn.Module):
    def __init__(self, camera_infos: list[CameraInfo]):
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
        fovY = fovs[:, 0]
        fovX = fovs[:, 1]
        tanHalfFovY = np.tan(fovY / 2)
        tanHalfFovX = np.tan(fovX / 2)

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
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
        # compu




if __name__ == "__main__":
    
    from colmap_reader import read_colmap_scene_info
    from viz_colmap import create_visual_scene
    import trimesh

    scene = read_colmap_scene_info("data/playroom")

    model = GaussianModel(scene)

    render = Render(scene.cameras)

    render(torch.tensor([0, 1, 3]), model)
