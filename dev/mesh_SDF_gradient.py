from mesh_to_sdf import mesh_to_sdf

import trimesh
import pyrender
import numpy as np

sm = trimesh.creation.box(np.array([5.0, 5.0, 5.0]))
sm.visual.vertex_colors = [0.0, 1.0, 0.0, 0.1]

bounds = [10, 10, 10]
increment = 0.5

x = np.arange(-bounds[0]/2, bounds[0]/2, increment)
y = np.arange(-bounds[1]/2, bounds[1]/2, increment)
z = np.arange(-bounds[2]/2, bounds[2]/2, increment)

sdf_points = []

for xi in x:
    for yi in y:
        for zi in z:
            sdf_points.append([xi, yi, zi])

sdf_points = np.array(sdf_points)

sdf = mesh_to_sdf(sm, sdf_points)

# colors = np.zeros(sdf_points.shape)
# colors[sdf < 0, 2] = 1
# colors[sdf > 0, 0] = 1
# cloud = pyrender.Mesh.from_points(sdf_points, colors=colors)
# scene = pyrender.Scene()
# scene.add(cloud)
# viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5)

sdf_grad = np.gradient(sdf, x, y, z) # This gives the volume normals

# Check a few positions
# We expect the normals here to be perpendicular to the cube faces

def find_nearest_ind(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_inds(x, y, z, p):
    inds = [0, 0, 0]
    inds[0] = find_nearest_ind(x,p[0])
    inds[1] = find_nearest_ind(y,p[1])
    inds[2] = find_nearest_ind(z,p[2])
    return inds

def get_grads(x, y, z, sdf_grad, p):
    ind = get_inds(x, y, z, p)
    grad = [0, 0, 0]
    grad[0] = sdf_grad[0][ind]
    grad[1] = sdf_grad[1][ind]
    grad[2] = sdf_grad[2][ind]
    return grad

# p = [2, 2, 2]
# print(get_grads(x, y, z, sdf_grad, p))
