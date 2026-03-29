from gs_model import GaussianModel
import os
from colmap_reader import read_colmap_scene_info
from viz_colmap import create_visual_scene
from render import Render

import trimesh
import torch


scene = read_colmap_scene_info("data/playroom")
model = GaussianModel(scene)

trimesh_scene = create_visual_scene(scene)
# trimesh_scene.show()


renderer = Render(scene.cameras)

renderer.forward(torch.tensor([0, 1, 3]), model)

print("Number of Gaussians: ", model._xyz.shape[0])