from gs_model import GaussianModel
import os
from colmap_reader import read_colmap_scene_info
from viz_colmap import create_visual_scene
from render import Render

import trimesh

model = GaussianModel()

scene = read_colmap_scene_info("data/playroom")

# trimesh_scene = create_visual_scene(scene)
# trimesh_scene.show()


model.create_from_pcd(scene)

renderer = Render(scene.cameras)

print("Number of Gaussians: ", model._xyz.shape[0])