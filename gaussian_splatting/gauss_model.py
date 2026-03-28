import torch
import torch.nn  as nn
import numpy as np
import math
from gaussian_splatting.utils.point_utils import PointCloud
from gaussian_splatting.gauss_render import strip_symmetric, inverse_sigmoid, build_scaling_rotation
from gaussian_splatting.utils.sh_utils import RGB2SH


def _compute_nn_sq_distances(pts):
    """Pure PyTorch nearest-neighbor squared distances (replaces simple_knn distCUDA2)."""
    device = pts.device
    N = pts.shape[0]
    chunk = 512
    nn_sq_dists = torch.zeros(N, device=device)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        dists = torch.cdist(pts[i:end], pts)          # (chunk, N) L2 distances
        rows = torch.arange(end - i, device=device)
        cols = torch.arange(i, end, device=device)
        dists[rows, cols] = float('inf')              # mask self-distance
        nn_sq_dists[i:end] = dists.min(dim=1).values ** 2
    return nn_sq_dists

class GaussModel(nn.Module):
    """
    A Gaussian Model

    * Attributes
    _xyz: locations of gaussians
    _feature_dc: DC term of features
    _feature_rest: rest features
    _rotatoin: rotation of gaussians
    _scaling: scaling of gaussians
    _opacity: opacity of gaussians

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    
    def __init__(self, sh_degree : int=3, debug=False):
        super(GaussModel, self).__init__()
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()
        self.debug = debug

    def create_from_pcd(self, pcd: PointCloud, device='cuda'):
        """Create the Gaussian model from a color point cloud."""
        points = pcd.coords
        colors = pcd.select_channels(['R', 'G', 'B']) / 255.

        fused_point_cloud = torch.tensor(np.asarray(points)).float().to(device)
        fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().to(device))

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(_compute_nn_sq_distances(fused_point_cloud), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        if self.debug:
            opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device=device)
        return self

    @classmethod
    def create_from_state_dict(cls, state_dict, device='cuda'):
        """Reconstruct model from a saved state_dict (for inference without the original point cloud)."""
        n_rest = state_dict['_features_rest'].shape[1]
        sh_degree = int(math.sqrt(n_rest + 1)) - 1
        N = state_dict['_xyz'].shape[0]

        model = cls(sh_degree=sh_degree)
        model._xyz          = nn.Parameter(torch.zeros(N, 3))
        model._features_dc  = nn.Parameter(torch.zeros(N, 1, 3))
        model._features_rest = nn.Parameter(torch.zeros(N, n_rest, 3))
        model._scaling      = nn.Parameter(torch.zeros(N, 3))
        model._rotation     = nn.Parameter(torch.zeros(N, 4))
        model._opacity      = nn.Parameter(torch.zeros(N, 1))
        model.max_radii2D   = torch.zeros(N)
        model.load_state_dict(state_dict)
        return model.to(device)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def save_ply(self, path):
        from plyfile import PlyData, PlyElement
        # import os
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l