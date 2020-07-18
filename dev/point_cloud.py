import trimesh
import pyrender
import numpy as np
import time

# tm = trimesh.creation.cylinder(radius=2, height=5)
# m = pyrender.Mesh.from_trimesh(tm)

# pts = tm.vertices.copy()
# colors = np.random.uniform(size=pts.shape)
# m = pyrender.Mesh.from_points(pts, colors=colors)

scene = pyrender.Scene()
v = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

for i in range(1,100):
    tm = trimesh.load("../viz_results/3D/new/frame_{:0>6d}.ply".format(i))
    pts = tm.vertices.copy()
    sm = trimesh.creation.uv_sphere(radius=0.2)
    sm.visual.vertex_colors = [1.0, 0.0, 0.0]
    tfs = np.tile(np.eye(4), (len(pts), 1, 1))
    tfs[:,:3,3] = pts
    m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    v.render_lock.acquire()
    scene.clear()
    scene.add(m)
    v.render_lock.release()

v.close_external()
while v.is_active:
    pass


