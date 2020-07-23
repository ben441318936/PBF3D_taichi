import trimesh
import pyrender
import numpy as np
import time

# pts = np.load("../viz_results/3D/new/npy/frame_199.npy")

# sm = trimesh.creation.uv_sphere(radius=0.2)
# sm.visual.vertex_colors = [1.0, 0.0, 0.0]
# tfs = np.tile(np.eye(4), (len(pts), 1, 1))
# tfs[:,:3,3] = pts
# m = pyrender.Mesh.from_trimesh(sm, poses=tfs)

# scene = pyrender.Scene()
# scene.add(m)

# v = pyrender.Viewer(scene, use_raymond_lighting=True, cull_faces=False)


# exp = 5

# prefix = "./meshing/exp{}/".format(exp)

# scene = pyrender.Scene()

# vertices = np.load(prefix+"vertices.npy") / 100
# faces = np.load(prefix+"faces.npy")
# normals = np.load(prefix+"normals.npy")
# tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# tm.visual.vertex_colors = np.zeros(shape=(tm.vertices.shape[0],4))
# tm.visual.vertex_colors[:,0] = 255
# tm.visual.vertex_colors[:,3] = 255

# m = pyrender.Mesh.from_trimesh(tm)

# scene.add(m)

# v = pyrender.Viewer(scene, use_raymond_lighting=True, cull_faces=False)


