import trimesh
import pyrender
import numpy as np

# tm = trimesh.creation.cylinder(radius=2, height=5)
# m = pyrender.Mesh.from_trimesh(tm)

# pts = tm.vertices.copy()
# colors = np.random.uniform(size=pts.shape)
# m = pyrender.Mesh.from_points(pts, colors=colors)

tm = trimesh.load("examples/models/frame_000590.ply")
pts = tm.vertices.copy()

sm = trimesh.creation.uv_sphere(radius=0.1)
sm.visual.vertex_colors = [1.0, 0.0, 0.0]
tfs = np.tile(np.eye(4), (len(pts), 1, 1))
tfs[:,:3,3] = pts
m = pyrender.Mesh.from_trimesh(sm, poses=tfs)

scene = pyrender.Scene()
scene.add(m)
pyrender.Viewer(scene, use_raymond_lighting=True)