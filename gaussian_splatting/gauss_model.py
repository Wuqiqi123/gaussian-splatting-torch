import torch
import torch.nn as nn
import numpy as np
from gaussian_splatting.utils.point_utils import PointCloud
from gaussian_splatting.gauss_render import strip_symmetric, inverse_sigmoid, build_scaling_rotation
from gaussian_splatting.color_network import HashColorNetwork


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
    3D Gaussian Splatting model.

    Geometry parameters (learnable):
        _xyz      [N, 3]  world-space positions
        _scaling  [N, 3]  log-space scale
        _rotation [N, 4]  unit quaternions
        _opacity  [N, 1]  pre-sigmoid opacity

    Appearance:
        color_net  HashColorNetwork  (hash-grid + tiny MLP, replaces SH)
    """

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return strip_symmetric(actual_covariance)

        self.scaling_activation         = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation      = build_covariance_from_scaling_rotation
        self.opacity_activation         = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation        = torch.nn.functional.normalize

    def __init__(self, debug=False):
        super().__init__()
        self._xyz      = torch.empty(0)
        self._scaling  = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity  = torch.empty(0)
        self.color_net = None
        self.setup_functions()
        self.debug = debug

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def create_from_pcd(self, pcd: PointCloud, device='cuda'):
        """Initialise from a colour point cloud."""
        points = pcd.coords

        xyz = torch.tensor(np.asarray(points)).float().to(device)
        print("Number of points at initialisation:", xyz.shape[0])

        dist2     = torch.clamp_min(_compute_nn_sq_distances(xyz), 1e-7)
        scales    = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots      = torch.zeros((xyz.shape[0], 4), device=device)
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones(xyz.shape[0], 1, device=device))

        if self.debug:
            opacities = inverse_sigmoid(0.9 * torch.ones_like(opacities))

        self._xyz      = nn.Parameter(xyz.requires_grad_(True))
        self._scaling  = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity  = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros(xyz.shape[0], device=device)

        self.color_net = HashColorNetwork().to(device)
        self.color_net.set_scene_bounds(xyz)
        return self

    @classmethod
    def create_from_state_dict(cls, state_dict, device='cuda'):
        """Reconstruct model from a saved checkpoint (no point cloud needed)."""
        N     = state_dict['_xyz'].shape[0]
        model = cls()
        model._xyz      = nn.Parameter(torch.zeros(N, 3))
        model._scaling  = nn.Parameter(torch.zeros(N, 3))
        model._rotation = nn.Parameter(torch.zeros(N, 4))
        model._opacity  = nn.Parameter(torch.zeros(N, 1))
        model.max_radii2D = torch.zeros(N)
        model.color_net = HashColorNetwork()
        model.load_state_dict(state_dict)
        return model.to(device)

    def get_param_groups(self):
        """
        Two learning-rate groups:
          - Gaussian geometry params  →  lr 1e-3
          - Hash-grid colour network  →  lr 1e-2  (hash grid benefits from higher lr)
        """
        return [
            {'params': [self._xyz, self._scaling, self._rotation, self._opacity], 'lr': 1e-3},
            {'params': list(self.color_net.parameters()), 'lr': 1e-2},
        ]

    # ------------------------------------------------------------------
    # Accessors (activations applied here, not stored)
    # ------------------------------------------------------------------

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
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
