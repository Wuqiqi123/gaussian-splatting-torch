from gs_model import GaussianModel
import os
from colmap_reader import read_colmap_scene_info

model = GaussianModel()




scene = read_colmap_scene_info("data/playroom")


model.create_from_pcd()