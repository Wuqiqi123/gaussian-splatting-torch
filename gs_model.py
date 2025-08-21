#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from torch import nn
import os
from scipy.spatial import KDTree
import torch
from sh_rgb import RGB2SH
from colmap_reader import SceneInfo
import pypose
import math

def build_scaling_rotation(s, R):
    S = torch.diag_embed(s)  
    L = R @ L 
    return L

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    '''Cov = L @ L^T = R @ S @ S^T @ R^T
    '''
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    covariance = L @ L.transpose(1, 2)
    return covariance

def inverse_sigmoid(x):
    return torch.log(x/(1-x))


class GaussianModel(nn.Module):
    def __init__(self, scene: SceneInfo,  sh_degree=3):
        super(GaussianModel, self).__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.percent_dense = 0

        self.spatial_lr_scale = scene.nerf_normalization["radius"]
        fused_point_cloud = torch.tensor(scene.point_cloud.points).float()
        fused_color = RGB2SH(torch.tensor(scene.point_cloud.colors).float())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(self.dist_kdtree(scene.point_cloud.points).float(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        rots = pypose.identity_SO3(fused_point_cloud.shape[0])

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))
        self.xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.scaling_logits = nn.Parameter(scales.requires_grad_(True))
        self.rotation = nn.Parameter(rots.requires_grad_(True))
        self.opacity_logits = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.xyz.shape[0]))
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(scene.cameras)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4)[None].repeat(len(scene.cameras), 1, 1)
        self.exposure = nn.Parameter(exposure.requires_grad_(True))

    @property
    def scaling(self):
        return torch.exp(self.scaling_logit)
    
    @property
    def features(self):
        features_dc = self.features_dc
        features_rest = self.features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    

    @property
    def opacity(self):
        return torch.sigmoid(self.opacity_logits)

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def covariance(self, scaling_modifier = 1):
        return build_covariance_from_scaling_rotation(self.scaling, scaling_modifier, self.rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def dist_kdtree(self, points_np):
        dists, inds = KDTree(points_np).query(points_np, k=4)
        meanDists = (dists[:, 1:] ** 2).mean(1)

        return torch.from_numpy(meanDists)


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1))
        self.denom = torch.zeros((self.get_xyz.shape[0], 1))

        l = [
            {'params': [self.xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self.features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self.features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self.opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self.rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)
        
        # self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
        #                                                 lr_delay_steps=training_args.exposure_lr_delay_steps,
        #                                                 lr_delay_mult=training_args.exposure_lr_delay_mult,
        #                                                 max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.features_dc.shape[1]*self.features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.features_rest.shape[1]*self.features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def reset_opacity(self):
        opacities_new = self.inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.xyz = optimizable_tensors["xyz"]
        self.features_dc = optimizable_tensors["f_dc"]
        self.features_rest = optimizable_tensors["f_rest"]
        self.opacity = optimizable_tensors["opacity"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling" : new_scaling,
             "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.xyz = optimizable_tensors["xyz"]
        self.features_dc = optimizable_tensors["f_dc"]
        self.features_rest = optimizable_tensors["f_rest"]
        self.opacity = optimizable_tensors["opacity"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = self.rotation[selected_pts_mask].repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self.rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self.features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self.opacity_logits[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self.xyz[selected_pts_mask]
        new_features_dc = self.features_dc[selected_pts_mask]
        new_features_rest = self.features_rest[selected_pts_mask]
        new_opacities = self.opacity_logits[selected_pts_mask]
        new_scaling = self.scaling[selected_pts_mask]
        new_rotation = self.rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


if __name__ == "__main__":
    from colmap_reader import read_colmap_scene_info
    from viz_colmap import create_visual_scene
    import trimesh

    scene = read_colmap_scene_info("data/playroom")

    model = GaussianModel(scene)

    print("11")