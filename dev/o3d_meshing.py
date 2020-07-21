import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("../viz_results/3D/new/frame_000100.ply")
print(pcd)

# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# radius = 3 * avg_dist

# bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

# dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

# dec_mesh.remove_degenerate_triangles()
# dec_mesh.remove_duplicated_triangles()
# dec_mesh.remove_duplicated_vertices()
# dec_mesh.remove_non_manifold_edges()

# o3d.visualization.draw_geometries([pcd, dec_mesh])

poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

o3d.visualization.draw_geometries([pcd, poisson_mesh])
