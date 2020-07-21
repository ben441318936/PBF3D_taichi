# import numpy as np
# import mcubes
# import open3d as o3d

# X, Y, Z = np.mgrid[:30, :30, :30]
# u = (X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2

# vertices, triangles = mcubes.marching_cubes(u, 0)
# mcubes.export_obj(vertices, triangles, 'sphere.obj')

# mesh = o3d.io.read_triangle_mesh("./sphere.obj")

# o3d.visualization.draw_geometries([mesh])

from marching_cubes import march
from numpy import load

volume = load("test/data/input/sample.npy")  # 128x128x128 uint8 volume

# extract the mesh where the values are larger than or equal to 1
# everything else is ignored
vertices, normals, faces = march(volume, 0)  # zero smoothing rounds
smooth_vertices, smooth_normals, faces = march(volume, 4)  # 4 smoothing rounds