import trimesh
import pyrender
import numpy as np

mesh = trimesh.load("heart2assem.PLY")
mesh.visual.vertex_colors = [0.0, 1.0, 0.0, 0.5]

pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
scene = pyrender.Scene()
scene.add(pyrender_mesh)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, show_world_axis=True)