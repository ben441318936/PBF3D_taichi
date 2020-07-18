import numpy as np
import mcubes
import open3d as o3d

X, Y, Z = np.mgrid[:30, :30, :30]
u = (X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2

vertices, triangles = mcubes.marching_cubes(u, 0)
mcubes.export_obj(vertices, triangles, 'sphere.obj')

mesh = o3d.io.read_triangle_mesh("./sphere.obj")

o3d.visualization.draw_geometries([mesh])