import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
import trimesh
import pyrender

scene = pyrender.Scene()
# Dummy node
sm = trimesh.creation.uv_sphere(radius=150)
sm.visual.vertex_colors = [1.0, 0.0, 0.0, 0.0]
tfs = np.tile(np.eye(4), (1, 1, 1))
tfs[:,:3,3] = np.array([[0,0,0]])
m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
nm = pyrender.Node(mesh=m, matrix=np.eye(4), name="mesh")
# Tool node
sm = trimesh.creation.box(np.array([10.0, 50.0, 10.0]))
sm.visual.vertex_colors = [0.0, 0.0, 1.0, 0.5]
tfs = np.tile(np.eye(4), (1, 1, 1))
tfs[:,:3,3] = np.array([[5,25,-5]])
m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
nt = pyrender.Node(mesh=m, matrix=np.eye(4), name="tool")
# Obstacle node
sm = trimesh.creation.box(np.array([100.0, 50.0, 50.0]))
sm.visual.vertex_colors = [0.0, 1.0, 0.0, 0.1]
tfs = np.tile(np.eye(4), (1, 1, 1))
tfs[:,:3,3] = np.array([[50,25,-25]])
m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
nb = pyrender.Node(mesh=m, matrix=np.eye(4), name="box")


scene.add_node(nm)
scene.add_node(nt)
scene.add_node(nb)
v = pyrender.Viewer(scene, use_raymond_lighting=True, cull_faces=False, run_in_thread=True, record=True)

print("Viewport size:", v.viewport_size)

exp = "exp32"

for k in range(0,300):
    if not v.is_active:
        break

    print("Preparing mesh {}".format(k))

    boundary = np.array([3.0, 3.0, 3.0])

    pts = np.load("../viz_results/3D/new_MPC/{}/particles/frame_{}.npy".format(exp,k))
    tool_pos = np.load("../viz_results/3D/new_MPC/{}/tool/frame_{}.npy".format(exp,k))

    if pts.shape[0] > 0:

        # # Use 3D Gaussian convolution to fill the volume from discrete points
        # # First prepare the whole environment as 3D grid
        # dx = 0.1
        # rounded_dims = np.round(boundary / dx).astype(int)
        # env = np.zeros((tuple(rounded_dims)))
        # # Put particles in discrete grid
        # inds = np.floor(pts / dx).astype(int)
        # for i in np.arange(inds.shape[0]):
        #     lim = env.shape
        #     if inds[i,0] < lim[0] and inds[i,1] < lim[1] and inds[i,2] < lim[2]:
        #         env[inds[i,0], inds[i,1], inds[i,2]] += 1

        # # Gaussian smoothing on the env
        # smooth_env = gaussian_filter(env, sigma=3)

        # padded_env = np.pad(smooth_env, 1, "constant", constant_values=0)

        # # Use marching cubes to obtain the surface mesh of these ellipsoids
        # vertices, faces, normals, values = measure.marching_cubes(padded_env, np.max(padded_env)/10)

        vertices = np.load("../viz_results/3D/new_MPC/{}/fluid/vertices_frame_{}.npy".format(exp,k))
        faces = np.load("../viz_results/3D/new_MPC/{}/fluid/faces_frame_{}.npy".format(exp,k))
        normals = np.load("../viz_results/3D/new_MPC/{}/fluid/normals_frame_{}.npy".format(exp,k))

        tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        # tm = trimesh.load_mesh("../viz_results/3D/new_MPC/fluid/frame_{}.ply".format(k))
        tm.visual.vertex_colors = np.zeros(shape=(tm.vertices.shape[0],4))
        tm.visual.vertex_colors[:,0] = 255
        tm.visual.vertex_colors[:,3] = 255

        m = pyrender.Mesh.from_trimesh(tm)

        v.render_lock.acquire()

        pose1 = np.eye(4)
        pose1[0:3,0:3] = np.array([[ 3.36824089e-01, -5.93911746e-02,  9.39692621e-01],
                                   [ 9.25416578e-01, -1.63175911e-01, -3.42020143e-01],
                                   [ 1.73648178e-01,  9.84807753e-01,  2.09426937e-17]])
        nm.mesh = m
        scene = v.scene
        # Set rotation pose for particles
        nodes = scene.get_nodes(name="mesh")
        for n in nodes:
            scene.set_pose(n, pose1)
        # Set translation and rotation for tool
        pose2 = np.eye(4)
        pose2[0:3,3] = tool_pos * 10 #np.array([120, 10, 100])
        pose2 = pose1 @ pose2
        nodes = scene.get_nodes(name="tool")
        for n in nodes:
            scene.set_pose(n, pose2)
        # Set rotation pose for box obstacle
        pose3 = np.eye(4)
        pose3[0:3,3] = 10 * np.array([0, 0, 10])
        pose3 = pose1 @ pose3
        nodes = scene.get_nodes(name="box")
        for n in nodes:
            scene.set_pose(n, pose3)

        # nodes = scene.camera_nodes
        # for n in nodes:
        #     print("Node transformation matrix:")
        #     print(n.matrix)
        #     cam = n.camera
        #     print("Camera settings")
        #     print("yfov:", cam.yfov)

        v.render_lock.release()

    else:
        break

print("Animation done")

v.close_external()

while v.is_active:
    pass

v.save_gif("./{}.gif".format(exp))