import taichi as ti
import numpy as np

ti.init(ti.cpu)

boundary = np.array([15, 20, 15])

inc = 0.5

x = np.arange(0, boundary[0], inc)
y = np.arange(0, boundary[1], inc)
z = np.arange(0, boundary[2], inc)

sdf_points = []

for xi in x:
    for yi in y:
        for zi in z:
            sdf_points.append([xi, yi, zi])

voxel_grid_dims = (x.shape[0], y.shape[0], z.shape[0])

box_sdf = ti.var(ti.f32)
box_sdf_grad = ti.Vector(3, ti.f32)

ti.root.dense(ti.ijk, voxel_grid_dims).place(box_sdf)
ti.root.dense(ti.ijk, voxel_grid_dims).place(box_sdf_grad)

box_sdf_array = np.load("box_sdf.npy")
print(box_sdf_array.shape)
box_sdf_grad_array = np.load("box_sdf_grad.npy")
print(box_sdf_grad_array.shape)

box_sdf.from_numpy(box_sdf_array)
box_sdf_grad.from_numpy(box_sdf_grad_array)

print(box_sdf[0,0,0])
print(box_sdf_grad[0,0,0][0], box_sdf_grad[0,0,0][1], box_sdf_grad[0,0,0][2])

@ti.func
def compute_voxel_ind(low, high, inc, v):
    if v <= low:
        return 0
    elif v >= high:
        return -1
    else:
        shifted_v = v - low
        return ti.cast(ti.floor(shifted_v / inc), ti.i32)

@ti.func
def confine_position_to_box_forward(p):
    ind = ti.Vector([0.0, 0.0, 0.0])
    ind[0] = compute_voxel_ind(0, boundary[0], inc, p[0])
    ind[1] = compute_voxel_ind(0, boundary[1], inc, p[1])
    ind[2] = compute_voxel_ind(0, boundary[2], inc, p[2])
    sdf_val = box_sdf[ind[0], ind[1], ind[2]]
    if sdf_val <= 0:
        normal = box_sdf_grad[ind[0], ind[1], ind[2]]
        p = p + -1 * sdf_val * normal
    return p

@ti.Kernel
def confine_pos(p: ti.ext_arr()):
    print(confine_position_to_box_forward(p))

confine_pos([2, 5, 9])