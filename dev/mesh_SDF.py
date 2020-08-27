'''
Uniform voxel
'''

# import numpy as np

# from mesh_to_sdf import mesh_to_voxels

# import trimesh
# from skimage import measure

# sm = trimesh.creation.box(np.array([10.0, 5.0, 5.0]))
# sm.visual.vertex_colors = [0.0, 1.0, 0.0, 0.1]

# voxels = mesh_to_voxels(sm, 128, pad=True)

# vertices, faces, normals, _ = measure.marching_cubes(voxels, level=0)
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# mesh.show()

'''
Points in space, more near surface
'''

# from mesh_to_sdf import sample_sdf_near_surface

# import trimesh
# import pyrender
# import numpy as np

# sm = trimesh.creation.box(np.array([10.0, 5.0, 5.0]))
# sm.visual.vertex_colors = [0.0, 1.0, 0.0, 0.1]

# points, sdf = sample_sdf_near_surface(sm, number_of_points=50000)

# colors = np.zeros(points.shape)
# colors[sdf < 0, 2] = 1
# colors[sdf > 0, 0] = 1
# cloud = pyrender.Mesh.from_points(points, colors=colors)
# scene = pyrender.Scene()
# scene.add(cloud)
# viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=3)

'''
Points in space, manually picked
If choose a regular grid, then this is like voxel grid
'''

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

print(sdf.shape)

colors = np.zeros(sdf_points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1
cloud = pyrender.Mesh.from_points(sdf_points, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5)