import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
import trimesh
import pyrender

scene = pyrender.Scene()
# Dummy node
sm = trimesh.creation.uv_sphere(radius=500)
sm.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]
tfs = np.tile(np.eye(4), (1, 1, 1))
tfs[:,:3,3] = np.array([[0,0,0]])
m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
nm = pyrender.Node(mesh=m, matrix=np.eye(4))

scene.add_node(nm)
v = pyrender.Viewer(scene, use_raymond_lighting=True, cull_faces=False, run_in_thread=True)

for k in range(200):
    print("Preparing new mesh")

    boundary = np.array([20.0, 20.0, 20.0])

    pts = np.load("../viz_results/3D/new/npy/frame_{}.npy".format(k))

    if pts.shape[0] > 0:

        # Use 3D Gaussian convolution to fill the volume from discrete points
        # First prepare the whole environment as 3D grid
        dx = 0.1
        rounded_dims = np.round(boundary / dx).astype(int)
        env = np.zeros((tuple(rounded_dims)))
        # Put particles in discrete grid
        inds = np.round(pts / dx).astype(int)
        for i in np.arange(inds.shape[0]):
            env[inds[i,0], inds[i,1], inds[i,2]] = 1

        # Gaussian smoothing on the env
        smooth_env = gaussian_filter(env, sigma=3)

        padded_env = np.pad(smooth_env, 1, "constant", constant_values=0)

        # Use marching cubes to obtain the surface mesh of these ellipsoids
        vertices, faces, normals, values = measure.marching_cubes(padded_env, np.max(padded_env)/8)

        tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        tm.visual.vertex_colors = np.zeros(shape=(tm.vertices.shape[0],4))
        tm.visual.vertex_colors[:,0] = 255
        tm.visual.vertex_colors[:,3] = 255

        m = pyrender.Mesh.from_trimesh(tm)

        v.render_lock.acquire()
        nm.mesh = m
        v.render_lock.release()

print("Animation done")

while v.is_active:
    pass