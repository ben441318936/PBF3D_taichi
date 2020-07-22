import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
import trimesh
import pyrender

exp = 4
prefix = "./meshing/exp{}/".format(exp)

boundary = np.array([40.0, 40.0, 40.0])

pts = np.load("../viz_results/3D/new/npy/frame_199.npy")
print(pts.shape)

# Use 3D Gaussian convolution to fill the volume from discrete points
# First prepare the whole environment as 3D grid
dx = 0.1
rounded_dims = np.round(boundary / dx).astype(int)
env = np.zeros((tuple(rounded_dims)))
# Put particles in discrete grid
inds = np.round(pts / dx).astype(int)
for i in range(inds.shape[0]):
    env[inds[i,0], inds[i,1], inds[i,2]] = 1

# Gaussian smoothing on the env
smooth_env = gaussian_filter(env, sigma=3)
smooth_env[smooth_env > np.max(smooth_env)/5] = 1
print(smooth_env.shape)

padded_env = np.pad(smooth_env, 1, "constant", constant_values=0)
print(padded_env.shape)

# Use marching cubes to obtain the surface mesh of these ellipsoids
vertices, faces, normals, values = measure.marching_cubes(padded_env, 0.5)

np.save(prefix+"vertices.npy", vertices)
np.save(prefix+"normals.npy", normals)
np.save(prefix+"faces.npy", faces)

tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
tm.visual.vertex_colors = np.zeros(shape=(tm.vertices.shape[0],4))
tm.visual.vertex_colors[:,0] = 255
tm.visual.vertex_colors[:,3] = 255

m = pyrender.Mesh.from_trimesh(tm)

scene = pyrender.Scene()
scene.add(m)
v = pyrender.Viewer(scene, use_raymond_lighting=True, cull_faces=False)