import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

from scipy.ndimage import gaussian_filter

exp = 3
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

# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = measure.marching_cubes(smooth_env, 0.5)

np.save(prefix+"vertices.npy", verts)
np.save(prefix+"normals.npy", normals)
np.save(prefix+"faces.npy", faces)

# # Display resulting triangular mesh using Matplotlib. This can also be done
# # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Fancy indexing: `verts[faces]` to generate a collection of triangles
# mesh = Poly3DCollection(verts[faces])
# mesh.set_edgecolor('k')
# ax.add_collection3d(mesh)

# ax.set_xlabel("x-axis: a = 6 per ellipsoid")
# ax.set_ylabel("y-axis: b = 10")
# ax.set_zlabel("z-axis: c = 16")

# ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
# ax.set_ylim(0, 20)  # b = 10
# ax.set_zlim(0, 32)  # c = 16

# plt.tight_layout()
# plt.show()