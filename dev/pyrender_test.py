import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
import trimesh
import pyrender


boundary = np.array([20.0, 20.0, 20.0])

pts = np.load("../viz_results/3D/new/npy/frame_199.npy")

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
# pc = pyrender.PerspectiveCamera(yfov=1.047, zfar=3029.348, znear=0.05, aspectRatio=None)
nm = pyrender.Node(mesh=m, matrix=np.eye(4), name="mesh")

# trans = np.array([[ 0.00000000e+00, -7.07106781e-01,  7.07106781e-01,  3.62901172e+02],
#                   [ 1.00000000e+00,  0.00000000e+00, -5.55111512e-17,  5.31364471e+01],
#                   [ 5.55111512e-17,  7.07106781e-01,  7.07106781e-01,  3.62855898e+02],
#                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
# nc = pyrender.Node(camera=pc, matrix=np.eye(4))

scene = pyrender.Scene()
scene.add_node(nm)
# scene.add_node(nc)

v = pyrender.Viewer(scene, use_raymond_lighting=True, cull_faces=False, show_world_axis=True, run_in_thread=True)

v.render_lock.acquire()
scene = v.scene
# nodes = scene.camera_nodes
nodes = scene.get_nodes(name="mesh")
for n in nodes:
    print(n.name)
    pose = np.eye(4)
    pose[0:3,0:3] = np.array([[ 3.36824089e-01, -5.93911746e-02,  9.39692621e-01],
                              [ 9.25416578e-01, -1.63175911e-01, -3.42020143e-01],
                              [ 1.73648178e-01,  9.84807753e-01,  2.09426937e-17]])
    scene.set_pose(n, pose)

# scene = v.scene
# nodes = scene.camera_nodes
# for n in nodes:
#     # cam = n.camera
#     # print("Camera intrinsics")
#     # print("Aspect ratio:", cam.aspectRatio)
#     # print("yfov:", cam.yfov)
#     # print("zfar:", cam.zfar)
#     # print("znear:", cam.znear)
#     print("Node transformation")
#     print("Matrix:")
#     print(n.matrix)

v.render_lock.release()



while v.is_active:
    pass

