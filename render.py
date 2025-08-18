from torch import nn
import torch
import numpy as np
from pypose import SE3, mat2SE3
from colmap_reader import CameraInfo
from gs_model import GaussianModel


class Render(nn.Module):
    def __init__(self, camera_infos: list[CameraInfo]):
        super(Render, self).__init__()
        self.uids = torch.stack([torch.tensor(cam_info.uid) for cam_info in camera_infos], dim=0)
        self.tf_world_cams = mat2SE3(torch.stack([torch.tensor(cam_info.tf_world_cam) for cam_info in camera_infos], dim=0).float())
        self.fovs = torch.stack([torch.tensor(cam_info.fov) for cam_info in camera_infos], dim=0)
        self.image_names = [cam_info.image_name for cam_info in camera_infos]
        self.gt_images = torch.stack([torch.tensor(cam_info.image) for cam_info in camera_infos], dim=0)

        self.zfar = 100.0
        self.znear = 0.01

        self.tf_cams_world = self.tf_world_cams.Inv()
        self.projection_matrix = self.get_projection_matrix(znear=self.znear, zfar=self.zfar, fovs=self.fovs)
        self.full_proj_transform = self.projection_matrix @ self.tf_cams_world.matrix()

    def get_projection_matrix(self, znear, zfar, fovs):
        fovY = fovs[:, 0]
        fovX = fovs[:, 1]
        tanHalfFovY = np.tan((fovY / 2))
        tanHalfFovX = np.tan((fovX / 2))

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

    def forward(self, indexes, gaussian_model: GaussianModel):
        pass