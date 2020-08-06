import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
import trimesh
import pyrender

exp = "exp6"

for k in range(0,177):

    print("Preparing mesh {}".format(k))

    boundary = np.array([20.0, 20.0, 20.0])

    pts = np.load("../viz_results/3D/new_MPC/{}/particles/frame_{}.npy".format(exp,k))
    tool_pos = np.load("../viz_results/3D/new_MPC/{}/tool/frame_{}.npy".format(exp,k))

    if pts.shape[0] > 0:

        # Use 3D Gaussian convolution to fill the volume from discrete points
        # First prepare the whole environment as 3D grid
        dx = 0.1
        rounded_dims = np.round(boundary / dx).astype(int)
        env = np.zeros((tuple(rounded_dims)))
        # Put particles in discrete grid
        inds = np.floor(pts / dx).astype(int)
        for i in np.arange(inds.shape[0]):
            lim = env.shape
            if inds[i,0] < lim[0] and inds[i,1] < lim[1] and inds[i,2] < lim[2]:
                env[inds[i,0], inds[i,1], inds[i,2]] += 1

        # Gaussian smoothing on the env
        smooth_env = gaussian_filter(env, sigma=3)

        padded_env = np.pad(smooth_env, 1, "constant", constant_values=0)

        # Use marching cubes to obtain the surface mesh of these ellipsoids
        vertices, faces, normals, values = measure.marching_cubes(padded_env, np.max(padded_env)/10)
        np.save("../viz_results/3D/new_MPC/{}/fluid/vertices_frame_{}.npy".format(exp,k), vertices)
        np.save("../viz_results/3D/new_MPC/{}/fluid/faces_frame_{}.npy".format(exp,k), faces)
        np.save("../viz_results/3D/new_MPC/{}/fluid/normals_frame_{}.npy".format(exp,k), normals)

        tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        tm.export("../viz_results/3D/new_MPC/{}/fluid/frame_{}.ply".format(exp,k))

    else:
        print("No points in file")
        break
