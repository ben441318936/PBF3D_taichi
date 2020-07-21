import numpy as np
import pyrender
import trimesh

from scipy.ndimage import gaussian_filter

from PIL import Image

exp = 2

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
smooth_env[smooth_env > np.max(smooth_env)/8] = 1
print(smooth_env.shape)

# Check discretization results with a cross section view
img_np = np.zeros((rounded_dims[0], rounded_dims[1], 3))
for i in range(3):
    img_np[:,:,i] = env[:,:,100] * 255
img_np = img_np = np.swapaxes(img_np,0,1)
img_np = img_np[::-1, :]
image = Image.fromarray(np.uint8(img_np),mode="RGB")
image.putalpha(255)
s = "./meshing/exp{}/env.png".format(exp)
image.save(s)

img_np = np.zeros((rounded_dims[0], rounded_dims[1], 3))
for i in range(3):
    img_np[:,:,i] = smooth_env[:,:,100] * 255
img_np = img_np = np.swapaxes(img_np,0,1)
img_np = img_np[::-1, :]
image = Image.fromarray(np.uint8(img_np),mode="RGB")
image.putalpha(255)
s = "./meshing/exp{}/smooth_env.png".format(exp)
image.save(s)

img_np = np.zeros((rounded_dims[0], rounded_dims[1], 3))
for i in range(inds.shape[0]):
    img_np[inds[i,0], inds[i,1],:] = 255
img_np = img_np = np.swapaxes(img_np,0,1)
img_np = img_np[::-1, :]
image = Image.fromarray(np.uint8(img_np),mode="RGB")
image.putalpha(255)
s = "./meshing/exp{}/side_view.png".format(exp)
image.save(s)

# Save volume as numpy array
np.save("./meshing/exp{}/volume.npy".format(exp), smooth_env)