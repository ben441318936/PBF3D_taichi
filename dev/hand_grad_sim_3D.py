import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

@ti.data_oriented
class HandGradSim3D:
    def __init__(self, max_timesteps=10, num_particles=10, do_emit=False, do_save_ply=False, do_save_npy=False, do_save_npz=False, do_render=False):

        self.dim = 3
        self.delta_t = 1.0 / 20.0
        self.max_timesteps = max_timesteps
        self.num_particles = num_particles

        self.do_emit = do_emit
        self.do_save_ply = do_save_ply
        self.do_save_npz = do_save_npz
        self.do_save_npy = do_save_npy
        self.do_render = do_render

        self.boundary = np.array([15.0, 20.0, 15.0])

        self.cell_size = 2.51
        self.cell_recpr = 1.0 / self.cell_size

        def round_up(f, s):
            return (math.floor(f * self.cell_recpr / s) + 1) * s

        self.grid_size = (round_up(self.boundary[0], 1), round_up(self.boundary[1], 1), round_up(self.boundary[2], 1))

        self.max_num_particles_per_cell = 200
        self.max_num_neighbors = 200

        self.epsilon = 1e-5
        self.particle_radius = 0.3

        # PBF params
        self.h = 1.1
        self.mass = 1.0
        self.rho0 = 2.0
        self.lambda_epsilon = 100.0
        self.vorticity_epsilon = 0.01
        self.viscosity_c = 0.1
        self.pbf_num_iters = 3
        self.corr_deltaQ_coeff = 0.3
        self.corrK = 0.001
        # Need ti.pow()
        # corrN = 4.0
        self.neighbor_radius = self.h * 1.05

        self.poly6_factor = 315.0 / 64.0 / np.pi
        self.spiky_grad_factor = -45.0 / np.pi

        self.inside_x_var = 0.5
        self.inside_z_var = 0.5
        self.gating_x_sig = 0.5
        self.gating_y_sig = 0.5
        self.gating_z_sig = 0.5


        self.particle_active = ti.var(ti.i32)
        self.num_active = ti.var(ti.i32)

        self.num_suctioned = ti.var(ti.i32)

        self.positions = ti.Vector(self.dim, ti.f32)
        self.positions_after_grav = ti.Vector(self.dim, ti.f32)
        self.positions_after_grav_box = ti.Vector(self.dim, ti.f32) # For obstacle after grav
        self.positions_after_delta = ti.Vector(self.dim, ti.f32)
        self.positions_after_delta_box = ti.Vector(self.dim, ti.f32) # For obstacle after delta
        self.positions_iter = ti.Vector(self.dim, ti.f32)

        self.velocities = ti.Vector(self.dim, ti.f32)

        self.grid_num_particles = ti.var(ti.i32)
        self.grid2particles = ti.var(ti.i32)
        self.particle_num_neighbors = ti.var(ti.i32)
        self.particle_neighbors = ti.var(ti.i32)
        
        self.lambdas = ti.var(ti.f32)
        self.position_deltas = ti.Vector(self.dim, dt=ti.f32)

        self.SGS_to_grad_j = ti.Vector(self.dim, ti.f32)
        self.local_grad_j = ti.Vector(self.dim, ti.f32)

        self.tool_states = ti.Vector(self.dim, dt=ti.f32)
        # 0: x, 1: y, 2: z 
        self.tool_dims = ti.Vector(self.dim, dt=ti.f32)

        self.voxel_inc = 0.1
        self.voxel_grid_dims = (len(np.arange(-1, self.boundary[0], self.voxel_inc)),
                                len(np.arange(-1, self.boundary[1], self.voxel_inc)),
                                len(np.arange(-1, self.boundary[2], self.voxel_inc)))

        self.box_sdf = ti.var(ti.f32)
        self.box_gradient = ti.Vector(3, ti.f32)
        self.box_laplacian = ti.Matrix(3, 3, ti.f32)

        self.loss = ti.var(ti.f32)

        self.place_vars()

    def place_vars(self):
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions, self.velocities)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions_after_grav)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions_after_grav_box)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions_after_delta)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions_after_delta_box)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.pbf_num_iters+1).dense(ti.k, self.num_particles).place(self.positions_iter)

        ti.root.place(self.loss)

        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.particle_active)
        ti.root.dense(ti.i, self.max_timesteps).place(self.num_active)
        ti.root.dense(ti.i, self.max_timesteps).place(self.num_suctioned)

        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.pbf_num_iters).dense(ti.k, self.num_particles).place(self.position_deltas)

        grid_snode = ti.root.dense(ti.i, self.max_timesteps).dense(ti.indices(1,2,3), self.grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.indices(4), self.max_num_particles_per_cell).place(self.grid2particles)

        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.pbf_num_iters).dense(ti.k, self.num_particles).place(self.lambdas)

        nb_node = ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.k, self.max_num_neighbors).place(self.particle_neighbors)

        ti.root.dense(ti.i, self.num_particles).dense(ti.j, self.max_num_neighbors).place(self.SGS_to_grad_j)
        ti.root.dense(ti.i, self.num_particles).dense(ti.j, self.max_num_neighbors).place(self.local_grad_j)

        ti.root.dense(ti.i, self.max_timesteps).place(self.tool_states)
        ti.root.place(self.tool_dims)

        ti.root.dense(ti.ijk, self.voxel_grid_dims).place(self.box_sdf)
        ti.root.dense(ti.ijk, self.voxel_grid_dims).place(self.box_gradient)
        ti.root.dense(ti.ijk, self.voxel_grid_dims).place(self.box_laplacian)

        ti.root.lazy_grad()

    @ti.kernel
    def init_tool(self, states: ti.ext_arr()):
        for i in range(self.max_timesteps):
            for k in ti.static(range(self.dim)):
                self.tool_states[i][k] = states[i,k]

    @ti.kernel
    def init_tool_dim(self, dims: ti.ext_arr()):
        self.tool_dims[None] = ti.Vector([dims[0], dims[1], dims[2]])

    def initialize(self, tool_states=None):
        self.positions.fill(0.0)
        self.positions_after_grav.fill(0.0)
        self.positions_after_grav_box.fill(0.0)
        self.positions_after_delta.fill(0.0)
        self.positions_after_delta_box.fill(0.0)
        self.positions_iter.fill(0.0)
        self.velocities.fill(0.0)
        self.lambdas.fill(0.0)
        self.position_deltas.fill(0.0)
        self.loss.fill(0.0)
        self.particle_active.fill(0)
        self.num_active.fill(0)
        self.num_suctioned.fill(0)
        self.clear_neighbor_info()
        # Temporary hardcoded values here
        self.tool_states.fill(100.0)
        self.init_tool_dim(np.array([1.0, 5.0, 1.0]))
        if tool_states is not None:
            self.init_tool(tool_states)
        self.box_sdf.from_numpy(np.load("box_sdf.npy"))
        self.box_gradient.from_numpy(np.load("box_sdf_gradient.npy"))
        self.box_laplacian.from_numpy(np.load("box_sdf_laplacian.npy"))

        
    def clear_global_grads(self):
        self.positions.grad.fill(0.0)
        self.positions_after_grav.grad.fill(0.0)
        self.positions_after_grav_box.grad.fill(0.0)
        self.positions_after_delta.grad.fill(0.0)
        self.positions_after_delta_box.grad.fill(0.0)
        self.positions_iter.grad.fill(0.0)
        self.velocities.grad.fill(0.0)
        self.lambdas.grad.fill(0.0)
        self.position_deltas.grad.fill(0.0)
        self.tool_states.grad.fill(0.0)
        self.loss.grad.fill(0.0)

    def clear_local_grads(self):
        self.SGS_to_grad_j.fill(0.0)
        self.local_grad_j.fill(0.0)

    @ti.kernel
    def place_particle(self, frame: ti.i32, i: ti.i32, p: ti.ext_arr(), v: ti.ext_arr()):
        pos = ti.Vector([p[0], p[1], p[2]])
        pos = self.confine_position_to_boundary_forward(pos)
        # pos = self.confine_position_to_tool_forward(i, pos, pos)
        for k in ti.static(range(self.dim)):
            self.positions[frame, i][k] = pos[k]
            self.velocities[frame, i][k] = v[k]

    def emit_particles(self, n, frame, p, v, p_status=None):
        for i in range(n):
            if self.num_active[frame] < self.num_particles:
                # offset = np.array([0,0,1])
                self.place_particle(frame, self.num_active[frame], p[i], v[i])
                if p_status is not None:
                    self.particle_active[frame, self.num_active[frame]] = p_status[i]
                else:
                    self.particle_active[frame, self.num_active[frame]] = 1
                self.num_active[frame] += 1

    @ti.func
    def confine_position_to_boundary_forward(self, p):
        # Global boundaries
        bmin = self.particle_radius
        bmax = ti.Vector([self.boundary[0], self.boundary[1], self.boundary[2]]) - self.particle_radius

        for i in ti.static(range(self.dim)):
            # Use randomness to prevent particles from sticking into each other after clamping
            if p[i] <= bmin:
                p[i] = bmin + self.epsilon * ti.random()
            elif bmax[i] <= p[i]:
                p[i] = bmax[i] - self.epsilon * ti.random()

        return p

    # @ti.func
    # def confine_position_to_box_forward(self, p):
    #     # Center obstacle
    #     box_pos = ti.Vector([0.0, 0.0, 10.0])
    #     box_dims = ti.Vector([10.0, 20.0, 5.0])

    #     box_front = box_pos[2]
    #     box_back = box_pos[2] - box_dims[2]
    #     box_left = box_pos[0]
    #     box_right = box_pos[0] + box_dims[0]
    #     box_bot = box_pos[1]
    #     box_top = box_pos[1] + box_dims[1]

    #     new_p = p

    #     if p[0] >= box_left and  p[0] <= box_right and p[1] >= box_bot and p[1] <= box_top and p[2] <= box_front and p[2] >= box_back:
    #         # d = ti.Vector([0.0, 0.0, 0.0])
    #         # d[0] = box_right - p[0]
    #         # d[1] = box_front - p[2] 
    #         # d[2] = p[2] - box_back

    #         # min_d = d[0]
    #         # ind = 0

    #         # for k in ti.static(range(3)):
    #         #     if d[k] <= min_d:
    #         #         ind = k

    #         # if ind == 0:
    #         #     new_p[0] = box_right + self.epsilon * ti.random()
    #         # if ind == 1:
    #         #     new_p[2] = box_front + self.epsilon * ti.random()
    #         # if ind == 2:
    #         #     new_p[2] = box_back - self.epsilon * ti.random()

    #         d_right = box_right - p[0]
    #         d_front = box_front - p[2]
    #         d_back = p[2] - box_back

    #         if d_right <= d_front and d_right <= d_back:
    #             new_p[0] = box_right + self.epsilon * ti.random()
    #         elif d_front <= d_right and d_front <= d_back:
    #             new_p[2] = box_front + self.epsilon * ti.random()
    #         else:
    #             new_p[2] = box_back - self.epsilon * ti.random()

    #     return new_p

    @ti.func
    def confine_position_to_boundary_backward(self, p):
        # Global boundaries
        bmin = self.particle_radius
        bmax = ti.Vector([self.boundary[0], self.boundary[1]+100, self.boundary[2]]) - self.particle_radius

        jacobian = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        for i in ti.static(range(self.dim)):
            if p[i] <= bmin or p[i] >= bmax[i]:
                jacobian[i,i] = 0

        return jacobian

    # @ti.func
    # def confine_position_to_box_backward(self, p):
    #     jacobian = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #     # Center obstacle
    #     box_pos = ti.Vector([0.0, 0.0, 10.0])
    #     box_dims = ti.Vector([10.0, 20.0, 5.0])

    #     box_front = box_pos[2]
    #     box_back = box_pos[2] - box_dims[2]
    #     box_left = box_pos[0]
    #     box_right = box_pos[0] + box_dims[0]
    #     box_bot = box_pos[1]
    #     box_top = box_pos[1] + box_dims[1]

    #     if p[0] >= box_left and  p[0] <= box_right and p[1] >= box_bot and p[1] <= box_top and p[2] <= box_front and p[2] >= box_back:
    #         # d = ti.Vector([0.0, 0.0, 0.0])
    #         # d[0] = ti.abs(p[0] - box_right)
    #         # d[1] = ti.abs(p[2] - box_front)
    #         # d[2] = ti.abs(p[2] - box_back)

    #         # min_d = d[0]
    #         # ind = 0

    #         # for k in ti.static(range(3)):
    #         #     if d[k] < min_d:
    #         #         ind = k    

    #         # if ind == 0:
    #         #     jacobian[0,0] = 0
    #         # elif ind == 1:
    #         #     jacobian[2,2] = 0
    #         # elif ind == 2:
    #         #     jacobian[2,2] = 0

    #         d_right = box_right - p[0]
    #         d_front = box_front - p[2]
    #         d_back = p[2] - box_back

    #         if d_right <= d_front and d_right <= d_back:
    #             jacobian[0,0] = 0
    #         elif d_front <= d_right and d_front <= d_back:
    #             jacobian[2,2] = 0
    #         else:
    #             jacobian[2,2] = 0

    #     return jacobian

    @ti.func
    def compute_voxel_ind(self, low, high, dim, inc, v):
        result = 0
        if v <= low:
            result = 0
        elif v >= high:
            result = dim-1
        else:
            shifted_v = v - low
            result = int(ti.floor(shifted_v / inc))
        return result

    @ti.func
    def confine_position_to_box_forward(self, p):
        new_p = p
        ind = ti.Vector([0, 0, 0])
        ind[0] = self.compute_voxel_ind(-1, self.boundary[0], self.voxel_grid_dims[0], self.voxel_inc, p[0])
        ind[1] = self.compute_voxel_ind(-1, self.boundary[1], self.voxel_grid_dims[1], self.voxel_inc, p[1])
        ind[2] = self.compute_voxel_ind(-1, self.boundary[2], self.voxel_grid_dims[2], self.voxel_inc, p[2])
        sdf_val = self.box_sdf[ind[0], ind[1], ind[2]]
        if sdf_val <= 0:
            normal = self.box_gradient[ind[0], ind[1], ind[2]]
            new_p = p + -1 * sdf_val * normal
        # print(p)
        # print(new_p)
        return new_p

    @ti.func
    def confine_position_to_box_backward(self, p):
        jacobian = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ind = ti.Vector([0, 0, 0])
        ind[0] = self.compute_voxel_ind(-1, self.boundary[0], self.voxel_grid_dims[0], self.voxel_inc, p[0])
        ind[1] = self.compute_voxel_ind(-1, self.boundary[1], self.voxel_grid_dims[1], self.voxel_inc, p[1])
        ind[2] = self.compute_voxel_ind(-1, self.boundary[2], self.voxel_grid_dims[2], self.voxel_inc, p[2])
        sdf_val = self.box_sdf[ind[0], ind[1], ind[2]]
        if sdf_val <= 0:
            normal = self.box_gradient[ind[0], ind[1], ind[2]]
            laplacian = self.box_laplacian[ind[0], ind[1], ind[2]]
            # d Dx / dx
            jacobian[0,0] += -1 * (normal[0] * normal[0] + sdf_val * laplacian[0,0])
            # d Dx / dy
            jacobian[1,0] += -1 * (normal[1] * normal[0] + sdf_val * laplacian[1,0])
            # d Dx / dz
            jacobian[2,0] += -1 * (normal[2] * normal[0] + sdf_val * laplacian[2,0])
            # d Dy / dx
            jacobian[0,1] += -1 * (normal[0] * normal[1] + sdf_val * laplacian[0,1])
            # d Dy / dy
            jacobian[1,1] += -1 * (normal[1] * normal[1] + sdf_val * laplacian[1,1])
            # d Dy / dz
            jacobian[2,1] += -1 * (normal[2] * normal[1] + sdf_val * laplacian[2,1])
            # d Dz / dx
            jacobian[0,2] += -1 * (normal[0] * normal[2] + sdf_val * laplacian[0,2])
            # d Dz / dy
            jacobian[1,2] += -1 * (normal[1] * normal[2] + sdf_val * laplacian[1,2])
            # d Dz / dz
            jacobian[2,2] += -1 * (normal[2] * normal[2] + sdf_val * laplacian[2,2])

        return jacobian
            

            
    @ti.kernel
    def gravity_forward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                g = ti.Vector([0.0, -9.81, 0.0])
                pos, vel = self.positions[frame-1,i], self.velocities[frame-1,i]
                vel += g * self.delta_t
                pos += vel * self.delta_t
                self.positions_after_grav[frame,i] = pos
                confined_pos = self.confine_position_to_boundary_forward(pos)
                self.positions_after_grav_box[frame, i] = confined_pos
                self.positions_iter[frame, 0, i] = self.confine_position_to_box_forward(confined_pos)
                # confined_pos = self.confine_position_to_box_forward(pos)
                # self.positions_after_grav_box[frame,i] = confined_pos
                # self.positions_iter[frame,0,i] = self.confine_position_to_boundary_forward(confined_pos)
            elif self.particle_active[frame,i] == 2:
                self.positions_iter[frame,0,i] = self.positions[frame-1,i]

    @ti.kernel
    def gravity_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1 or self.particle_active[frame-1,i] == 1:
                pos = self.positions_after_grav[frame, i]
                jacobian_bounds = self.confine_position_to_boundary_backward(pos)
                pos = self.positions_after_grav_box[frame, i]
                jacobian_box = self.confine_position_to_box_backward(pos)

                self.positions.grad[frame-1,i] += jacobian_box @ jacobian_bounds @ self.positions_iter.grad[frame,0,i]
                self.velocities.grad[frame-1,i] += jacobian_box @ jacobian_bounds @ self.positions_iter.grad[frame,0,i] * self.delta_t

            elif self.particle_active[frame,i] == 2:
                self.positions.grad[frame-1,i] += self.positions_iter.grad[frame,0,i]


    @ti.func
    def upward_field_forward(self, x, x0, z, z0):
        v = ti.Vector([0.0, 0.0, 0.0])
        v[1] = 100 * ((2*np.pi)**2 * self.inside_x_var * self.inside_z_var)**(-1/2) * ti.exp(-1/2 * ((x-x0)**2 / self.inside_x_var + (z-z0)**2 / self.inside_z_var))
        return v

    @ti.func
    def upward_field_backward(self, x, x0, z, z0):
        jacobian = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # d Dx / dx
        jacobian[0,0] = 0
        # d Dx / dy
        jacobian[1,0] = 0
        # d Dx / dz
        jacobian[2,0] = 0
        # d Dy / dx
        jacobian[0,1] = -1 * ((2*np.pi)**2 * self.inside_x_var * self.inside_z_var)**(-1/2) \
                        * ti.exp(-1/2 * ((x-x0)**2 / self.inside_x_var + (z-z0)**2 / self.inside_z_var)) \
                        * (x-x0) / self.inside_x_var
        # d Dy / dy
        jacobian[1,1] = 0
        # d Dy / dz
        jacobian[2,1] = -1 * ((2*np.pi)**2 * self.inside_x_var * self.inside_z_var)**(-1/2) \
                        * ti.exp(-1/2 * ((x-x0)**2 / self.inside_x_var + (z-z0)**2 / self.inside_z_var)) \
                        * (z-z0) / self.inside_z_var
        # d Dz / dx
        jacobian[0,2] = 0
        # d Dz / dy
        jacobian[1,2] = 0
        # d Dz / dz
        jacobian[2,2] = 0
        return jacobian * 100

    # Generic for all three dimensions
    @ti.func
    def gating_increase_forward(self, x, x0, sigma):
        return 1 / (1 + ti.exp(-1 * (x-x0) / sigma))

    @ti.func
    def gating_increase_backward(self, x, x0, sigma):
        return 1 / sigma * self.gating_increase_forward(x,x0,sigma) * (1 - self.gating_increase_forward(x,x0,sigma))

    @ti.func
    def gating_decrease_forward(self, x, x0, sigma):
        return 1 / (1 + ti.exp(1 * (x-x0) / sigma))

    @ti.func
    def gating_decrease_backward(self, x, x0, sigma):
        return -1 / sigma * self.gating_decrease_forward(x,x0,sigma) * (1 - self.gating_decrease_forward(x,x0,sigma))

    @ti.func
    def center_force_forward(self, pos, center):
        tmp = center - pos
        unit = tmp / tmp.norm()
        a = 4
        c = 2
        return a * unit / (tmp.norm()**2 + c)

    @ti.func
    def center_force_backward(self, pos, center):
        tmp = center - pos
        norm = tmp.norm()
        a = 4
        c = 2
        jacobian = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # d Dx / dx
        jacobian[0,0] = norm**(-1) * (norm**2+c)**(-1) \
                            - tmp[0]**2 * norm**(-3) * (norm**2+c)**(-1) \
                            - 2 * tmp[0]**2 * norm**(-1) * (norm**2+c)**(-2)
        # d Dx / dy
        jacobian[1,0] = -1 * tmp[0] * tmp[1] * norm**(-3) * (norm**2+c)**(-1) \
                            - 2 * tmp[0] * tmp[1] * norm**(-1) * (norm**2+c)**(-2)
        # d Dx / dz
        jacobian[2,0] = -1 * tmp[0] * tmp[2] * norm**(-3) * (norm**2+c)**(-1) \
                            - 2 * tmp[0] * tmp[2] * norm**(-1) * (norm**2+c)**(-2)

        # d Dy / dx
        jacobian[0,1] = -1 * tmp[1] * tmp[0] * norm**(-3) * (norm**2+c)**(-1) \
                            - 2 * tmp[1] * tmp[0] * norm**(-1) * (norm**2+c)**(-2)
        # d Dy / dy
        jacobian[1,1] = norm**(-1) * (norm**2+c)**(-1) \
                            - tmp[1]**2 * norm**(-3) * (norm**2+c)**(-1) \
                            - 2 * tmp[1]**2 * norm**(-1) * (norm**2+c)**(-2)
        # d Dy / dz
        jacobian[2,1] = -1 * tmp[1] * tmp[2] * norm**(-3) * (norm**2+c)**(-1) \
                            - 2 * tmp[1] * tmp[2] * norm**(-1) * (norm**2+c)**(-2)
        # d Dz / dx
        jacobian[0,2] = -1 * tmp[2] * tmp[0] * norm**(-3) * (norm**2+c)**(-1) \
                            - 2 * tmp[2] * tmp[0] * norm**(-1) * (norm**2+c)**(-2)
        # d Dz / dy
        jacobian[1,2] = -1 * tmp[2] * tmp[1] * norm**(-3) * (norm**2+c)**(-1) \
                            - 2 * tmp[2] * tmp[1] * norm**(-1) * (norm**2+c)**(-2)
        # d Dy / dy
        jacobian[2,2] = norm**(-1) * (norm**2+c)**(-1) \
                            - tmp[2]**2 * norm**(-3) * (norm**2+c)**(-1) \
                            - 2 * tmp[2]**2 * norm**(-1) * (norm**2+c)**(-2)

        return a * jacobian


    @ti.kernel
    def apply_suction_forward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos_i = self.positions_after_delta[frame,i]
                
                tool_states = self.tool_states[frame]
                dims = self.tool_dims[None]

                tool_left = tool_states[0]
                tool_right = tool_states[0] + dims[0]
                tool_bot = tool_states[1]
                tool_top = tool_states[1] + dims[1]
                tool_front = tool_states[2]
                tool_back = tool_states[2] - dims[2]

                tool_center = tool_states + ti.Vector([1*dims[0]/2, 0, -1*dims[2]/2])
                x0 = tool_center[0]
                y0 = tool_center[1]
                z0 = tool_center[2]

                delta = ti.Vector([0.0, 0.0, 0.0])
                x = pos_i[0]
                y = pos_i[1]
                z = pos_i[2]

                #Inside field
                delta += self.upward_field_forward(x, x0, z, z0) * self.gating_increase_forward(y, tool_bot-1, self.gating_y_sig)
                # Outside field
                delta += self.center_force_forward(pos_i, tool_center) \
                            * self.gating_increase_forward(x, tool_left-0.5, self.gating_x_sig) \
                            * self.gating_decrease_forward(x, tool_right+0.5, self.gating_x_sig) \
                            * self.gating_increase_forward(z, tool_back-0.5, self.gating_z_sig) \
                            * self.gating_decrease_forward(z, tool_front+0.5, self.gating_z_sig) \
                            * self.gating_increase_forward(y, tool_bot-1, self.gating_y_sig) \
                            * self.gating_decrease_forward(y, tool_bot+0.2, self.gating_y_sig) 
                
                self.positions[frame,i] = pos_i +  delta


    # This backward will propagate back to particle position and to tool position
    @ti.kernel
    def apply_suction_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos_i = self.positions_after_delta[frame,i]

                tool_states = self.tool_states[frame]
                dims = self.tool_dims[None]

                tool_left = tool_states[0]
                tool_right = tool_states[0] + dims[0]
                tool_bot = tool_states[1]
                tool_top = tool_states[1] + dims[1]
                tool_front = tool_states[2]
                tool_back = tool_states[2] - dims[2]
                
                tool_center = tool_states + ti.Vector([1*dims[0]/2, 0, -1*dims[2]/2])
                x0 = tool_center[0]
                y0 = tool_center[1]
                z0 = tool_center[2]

                x = pos_i[0]
                y = pos_i[1]
                z = pos_i[2]

                upstream_grad = self.positions.grad[frame,i]
                # print(upstream_grad)

                eye = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

                inside_grad = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

                # Inside field with gating
                upward_grad = self.upward_field_backward(x, x0, z, z0)
                upward_force = self.upward_field_forward(x, x0, z, z0)
                # d Dx / dx
                inside_grad[0,0] = 0
                # d Dx / dy
                inside_grad[1,0] = 0
                # d Dx / dz
                inside_grad[2,0] = 0
                # d Dy / dx
                inside_grad[0,1] = upward_grad[0,1] * self.gating_increase_forward(y, tool_bot-1, self.gating_y_sig)
                # d Dy / dy
                inside_grad[1,1] = upward_force[1] * self.gating_increase_backward(y, tool_bot-1, self.gating_y_sig)
                # d Dy / dz
                inside_grad[2,1] = upward_grad[2,1] * self.gating_increase_forward(y, tool_bot-1, self.gating_y_sig)
                # d Dz / dx
                inside_grad[0,2] = 0
                # d Dz / dy
                inside_grad[1,2] = 0
                # d Dz / dz
                inside_grad[2,2] = 0

                outside_grad = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

                # Outside field with gating
                center_force = self.center_force_forward(pos_i, tool_center)
                center_force_grad = self.center_force_backward(pos_i, tool_center)

                gate_x = self.gating_increase_forward(x, tool_left-0.5, self.gating_x_sig) * self.gating_decrease_forward(x, tool_right+0.5, self.gating_x_sig)
                gate_x_grad = self.gating_increase_backward(x, tool_left-0.5, self.gating_x_sig) * self.gating_decrease_forward(x, tool_right+0.5, self.gating_x_sig) \
                                * self.gating_increase_forward(x, tool_left-0.5, self.gating_x_sig) * self.gating_decrease_backward(x, tool_right+0.5, self.gating_x_sig)

                gate_y = self.gating_increase_forward(y, tool_bot-1, self.gating_y_sig) * self.gating_decrease_forward(y, tool_bot+0.2, self.gating_y_sig)
                gate_y_grad = self.gating_increase_backward(y, tool_bot-1, self.gating_y_sig) * self.gating_decrease_forward(y, tool_bot+0.2, self.gating_y_sig) \
                                * self.gating_increase_forward(y, tool_bot-1, self.gating_y_sig) * self.gating_decrease_backward(y, tool_bot+0.2, self.gating_y_sig)


                gate_z = self.gating_increase_forward(z, tool_back-0.5, self.gating_z_sig) * self.gating_decrease_forward(z, tool_front+0.5, self.gating_z_sig)
                gate_z_grad = self.gating_increase_backward(z, tool_back-0.5, self.gating_z_sig) * self.gating_decrease_forward(z, tool_front+0.5, self.gating_z_sig) \
                                * self.gating_increase_forward(z, tool_back-0.5, self.gating_z_sig) * self.gating_decrease_backward(z, tool_front+0.5, self.gating_z_sig)

                # d Dx / dx
                outside_grad[0,0] = center_force_grad[0,0] * gate_x * gate_y * gate_z \
                                    * center_force[0] * gate_x_grad * gate_y * gate_z
                # d Dx / dy
                outside_grad[1,0] = center_force_grad[1,0] * gate_x * gate_y * gate_z \
                                    * center_force[0] * gate_x * gate_y_grad * gate_z
                # d Dx / dz
                outside_grad[2,0] = center_force_grad[2,0] * gate_x * gate_y * gate_z \
                                    * center_force[0] * gate_x * gate_y * gate_z_grad
                # d Dy / dx
                outside_grad[0,1] = center_force_grad[0,1] * gate_x * gate_y * gate_z \
                                    * center_force[1] * gate_x_grad * gate_y * gate_z
                # d Dy / dy
                outside_grad[1,1] = center_force_grad[1,1] * gate_x * gate_y * gate_z \
                                    * center_force[1] * gate_x * gate_y_grad * gate_z
                # d Dy / dz
                outside_grad[2,1] = center_force_grad[2,1] * gate_x * gate_y * gate_z \
                                    * center_force[1] * gate_x * gate_y * gate_z_grad
                # d Dz / dx
                outside_grad[0,2] = center_force_grad[0,2] * gate_x * gate_y * gate_z \
                                    * center_force[2] * gate_x_grad * gate_y * gate_z
                # d Dz / dy
                outside_grad[1,2] = center_force_grad[1,2] * gate_x * gate_y * gate_z \
                                    * center_force[2] * gate_x * gate_y_grad * gate_z
                # d Dz / dz
                outside_grad[2,2] = center_force_grad[2,2] * gate_x * gate_y * gate_z \
                                    * center_force[2] * gate_x * gate_y * gate_z_grad

                pos_grad_delta = 1 * inside_grad - outside_grad
                tool_grad_delta = -1 * inside_grad + outside_grad

                self.positions_after_delta.grad[frame,i] += 1 * (eye + pos_grad_delta) @ upstream_grad
                self.tool_states.grad[frame] += 1 * tool_grad_delta @ upstream_grad


    # Propagating the resting particles is only needed in the aux sim for loss formulation
    @ti.kernel
    def prop_resting_forward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 2:
                self.positions[frame,i] = self.positions_iter[frame,0,i]
    
    @ti.kernel
    def prop_resting_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 2:
                self.positions_iter.grad[frame,0,i] = self.positions.grad[frame,i]
                        
    @ti.kernel
    def update_velocity_froward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                self.velocities[frame,i] = (self.positions[frame,i] - self.positions[frame-1,i]) / self.delta_t

    @ti.kernel
    def update_velocity_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                if frame == self.max_timesteps-1:
                    pass
                else:
                    self.positions.grad[frame,i] += self.velocities.grad[frame,i] / self.delta_t
                    self.positions.grad[frame-1,i] += - self.velocities.grad[frame,i] / self.delta_t

    @ti.func
    def get_cell(self, pos):
        return ti.cast(pos * self.cell_recpr, ti.i32)

    @ti.func
    def is_in_grid(self,c):
        # @c: Vector(i32)
        return 0 <= c[0] and c[0] < self.grid_size[0] and 0 <= c[1] and c[1] < self.grid_size[1] and 0 <= c[2] and c[2] < self.grid_size[2]

    @ti.kernel
    def update_grid(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                cell = self.get_cell(self.positions_iter[frame,0,i])
                # ti.Vector doesn't seem to support unpacking yet
                # but we can directly use int Vectors as indices
                offs = self.grid_num_particles[frame, cell].atomic_add(1)
                self.grid2particles[frame, cell, offs] = i

    @ti.kernel
    def find_particle_neighbors(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos_i = self.positions_iter[frame,0,i]
                cell = self.get_cell(pos_i)
                nb_i = 0
                for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                    cell_to_check = cell + offs
                    if self.is_in_grid(cell_to_check):
                        for j in range(self.grid_num_particles[frame, cell_to_check]):
                            p_j = self.grid2particles[frame, cell_to_check, j]
                            if nb_i < self.max_num_neighbors and p_j != i:
                                dist = (pos_i - self.positions_iter[frame,0,p_j]).norm()
                                if (dist < self.neighbor_radius):
                                    self.particle_neighbors[frame, i, nb_i] = p_j
                                    nb_i += 1
                self.particle_num_neighbors[frame, i] = nb_i

    # Output is a 2-vector
    @ti.func
    def spiky_gradient_forward(self, r, h):
        result = ti.Vector([0.0, 0.0, 0.0])
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = self.spiky_grad_factor * x * x
            result = r * g_factor / r_len
        return result

    # Upstream gradient is a 2-vector
    @ti.func
    def spiky_gradient_backward(self, r, h):
        jacobian = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        c = 1
        l = r.norm()
        if 0 < l and l < h:
            f = ti.static(self.spiky_grad_factor)
            # d Gx / dx 
            jacobian[0,0] = f / h**6 * ( \
                            (h-l)**2 / l \
                            - 2 * r[0]**2 * (h-l) / l**2 \
                            - r[0]**2 * (h-l)**2 / l**3)
            # d Gx / dy
            jacobian[1,0] = f / h**6 * ( \
                            - 2 * r[0] * r[1] * (h-l) / l**2 \
                            - r[0] * r[1] * (h-l)**2 / l**2)
            # d Gx / dz
            jacobian[2,0] = f / h**6 * ( \
                            - 2 * r[0] * r[2] * (h-l) / l**2 \
                            - r[0] * r[2] * (h-l)**2 / l**2)
            # d Gy / dx
            jacobian[0,1] = f / h**6 * ( \
                            - 2 * r[1] * r[0] * (h-l) / l**2 \
                            - r[1] * r[0] * (h-l)**2 / l**2)
            # d Gy / dy
            jacobian[1,1] = f / h**6 * ( \
                            (h-l)**2 / l \
                            - 2 * r[1]**2 * (h-l) / l**2 \
                            - r[1]**2 * (h-l)**2 / l**3)
            # d Gy / dz
            jacobian[2,1] = f / h**6 * ( \
                            - 2 * r[1] * r[2] * (h-l) / l**2 \
                            - r[1] * r[2] * (h-l)**2 / l**2)
            # d Gz / dx
            jacobian[0,2] = f / h**6 * ( \
                            - 2 * r[2] * r[0] * (h-l) / l**2 \
                            - r[2] * r[0] * (h-l)**2 / l**2)
            # d Gz / dy
            jacobian[1,2] = f / h**6 * ( \
                            - 2 * r[2] * r[1] * (h-l) / l**2 \
                            - r[2] * r[1] * (h-l)**2 / l**2) 
            # d Gy / dy
            jacobian[2,2] = f / h**6 * ( \
                            (h-l)**2 / l \
                            - 2 * r[2]**2 * (h-l) / l**2 \
                            - r[2]**2 * (h-l)**2 / l**3)       
        for i in ti.static(range(self.dim)):
            for j in ti.static(range(self.dim)):
                if jacobian[i,j] >= c:
                    jacobian[i,j] = c
                if jacobian[i,j] <= -c:
                    jacobian[i,j] = -c
        return jacobian

    # Use s^2 = |r|^2 for easier differentiation
    # Output is a scalar
    @ti.func
    def poly6_value_forward(self, s_sqr, h):
        result = 0.0
        #s_sqr = r.norm()**2
        if 0 < s_sqr and s_sqr < h**2:
            x = (h * h - s_sqr) / (h * h * h)
            result = self.poly6_factor * x * x * x
        return result
    
    # Upstream gradient is a scalar
    @ti.func
    def poly6_value_backward(self, s_sqr, h):
        #s_sqr = r.norm()**2
        out_grad = 0.0
        if 0 < s_sqr and s_sqr < h**2:
            out_grad = self.poly6_factor / h**9 * 3 * (h**2 - s_sqr) #* ti.Vector([1.0, 1.0])
        else:
            out_grad = 0.0
        return out_grad

    @ti.func
    def s_sqr_to_r(self, r):
        return 2 * r


    @ti.kernel
    def compute_lambdas_forward(self, frame: ti.i32, it: ti.i32):
        # Eq (8) ~ (11)
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos_i = self.positions_iter[frame,it,i]

                grad_i = ti.Vector([0.0, 0.0, 0.0])
                sum_gradient_sqr = 0.0
                density_constraint = 0.0

                for j in range(self.particle_num_neighbors[frame, i]):
                    p_j = self.particle_neighbors[frame, i, j]
                    if p_j >= 0:
                        pos_ji = pos_i - self.positions_iter[frame,it,p_j]
                        grad_j = self.spiky_gradient_forward(pos_ji, self.h)
                        grad_i += grad_j
                        sum_gradient_sqr += grad_j.dot(grad_j)
                        # Eq(2)
                        density_constraint += self.poly6_value_forward(pos_ji.norm()**2, self.h)

                # Eq(1)
                density_constraint = (self.mass * density_constraint / self.rho0) - 1.0

                sum_gradient_sqr += grad_i.dot(grad_i)
                self.lambdas[frame, it, i] = (-density_constraint) / (sum_gradient_sqr +
                                                        self.lambda_epsilon)

    @ti.kernel
    def compute_lambdas_backward(self, frame: ti.i32, it: ti.i32):
        # Gradient from lambda(i)
        # Here I am recomputing intermediate values
        for i in range(self.num_particles):
            # if self.particle_active[frame-1,i] == 1:
            pos_i = self.positions_iter[frame,it,i]
            grad_i = ti.Vector([0.0, 0.0, 0.0])
            sum_gradient_sqr = 0.0
            density_constraint = 0.0

            grad_to_lambda_i = self.lambdas.grad[frame,it,i]

            for j in range(self.particle_num_neighbors[frame, i]):
                p_j = self.particle_neighbors[frame, i, j]
                if p_j >= 0:
                    pos_ji = pos_i - self.positions_iter[frame,it,p_j]
                    grad_j = self.spiky_gradient_forward(pos_ji, self.h)
                    # Accumulate grad_j during recomputing
                    self.SGS_to_grad_j[i,j] = 2 * grad_j
                    grad_i += grad_j
                    sum_gradient_sqr += grad_j.dot(grad_j)
                    # Eq(2)
                    density_constraint += self.poly6_value_forward(pos_ji.norm()**2, self.h)
            # Eq(1)
            density_constraint = (self.mass * density_constraint / self.rho0) - 1.0
            sum_gradient_sqr += grad_i.dot(grad_i)

            lambda_to_SGS = -density_constraint / (sum_gradient_sqr + self.lambda_epsilon)**2
            lambda_to_constraint = -1 / (sum_gradient_sqr + self.lambda_epsilon)

            # There is contribution to pos_i from the spiky computation of each neighbor
            SGS_to_pos_i = ti.Vector([0.0, 0.0, 0.0])
            # There is contribution to pos_i from the poly6 computation of each neighbor
            constraint_to_pos_i = ti.Vector([0.0, 0.0, 0.0])

            constraint_to_val = -1 * self.mass / self.rho0

            for j in range(self.particle_num_neighbors[frame, i]):
                p_j = self.particle_neighbors[frame, i, j]
                if p_j >= 0:
                    pos_ji = pos_i - self.positions_iter[frame,it,p_j]

                    # SGS path the spiky gradients
                    self.SGS_to_grad_j[i,j] += 2 * grad_i
                    SGS_to_r_ji = self.spiky_gradient_backward(pos_ji, self.h) @ self.SGS_to_grad_j[i,j]
                    # For each neighbor pos_j, the contribution comes from one spiky grad computation
                    self.positions_iter.grad[frame,it,p_j] += -1 * SGS_to_r_ji * lambda_to_SGS * grad_to_lambda_i
                    # For every spiky gradient computation, there is a contribution to pos_i
                    SGS_to_pos_i += SGS_to_r_ji

                    # Density constraint path with poly6 values
                    constraint_to_r_ji = self.poly6_value_backward(pos_ji.norm()**2, self.h) * self.s_sqr_to_r(pos_ji) * constraint_to_val
                    # For each neighbor pos_j, the constribution comes from one poly6 val computation
                    self.positions_iter.grad[frame,it,p_j] += -1 * constraint_to_r_ji * lambda_to_constraint * grad_to_lambda_i
                    # For every poly6 value computation, there is a constribution to pos_i
                    constraint_to_pos_i += constraint_to_r_ji

            # Sum the contribution from the SGS and constraint paths
            self.positions_iter.grad[frame,it,i] += (SGS_to_pos_i * lambda_to_SGS \
                                                        + constraint_to_pos_i * lambda_to_constraint) * \
                                                            grad_to_lambda_i

    @ti.func
    def compute_scorr_forward(self, pos_ji):
        # Eq (13)
        x = self.poly6_value_forward(pos_ji.norm()**2, self.h) / self.poly6_value_forward((self.corr_deltaQ_coeff*self.h)**2, self.h)
        # pow(x, 4)
        x = x * x
        x = x * x
        return (-self.corrK) * x

    @ti.func
    def compute_scorr_backward(self, pos_ji):
        poly6_ji = self.poly6_value_forward(pos_ji.norm()**2, self.h)
        poly6_delta_Q = self.poly6_value_forward((self.corr_deltaQ_coeff*self.h)**2, self.h)
        poly6_grad = self.poly6_value_backward(pos_ji.norm()**2, self.h) * self.s_sqr_to_r(pos_ji)

        grad = -4 * self.corrK * (poly6_ji / poly6_delta_Q)**3 / poly6_delta_Q * poly6_grad
        return grad

    @ti.kernel
    def compute_position_deltas_forward(self, frame: ti.i32, it: ti.i32):
        # Eq(12), (14)
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos_i = self.positions_iter[frame,it,i]
                lambda_i = self.lambdas[frame,it,i]

                pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
                for j in range(self.particle_num_neighbors[frame,i]):
                    p_j = self.particle_neighbors[frame,i, j]
                    # TODO: does taichi supports break?
                    if p_j >= 0:
                        lambda_j = self.lambdas[frame,it,p_j]
                        pos_ji = pos_i - self.positions_iter[frame,it,p_j]
                        scorr_ij = self.compute_scorr_forward(pos_ji)
                        pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                            self.spiky_gradient_forward(pos_ji, self.h)

                pos_delta_i /= self.rho0
                self.position_deltas[frame,it,i] = pos_delta_i

    @ti.kernel
    def compute_position_deltas_backward(self, frame: ti.i32, it: ti.i32):
        for i in range(self.num_particles):
            # if self.particle_active[frame-1,i] == 1:
            pos_i = self.positions_iter[frame,it,i]
            lambda_i = self.lambdas[frame,it,i]

            grad_to_delta_i = self.position_deltas.grad[frame,it,i]
            # print("Frame", frame, "particle", i, grad_to_delta_i[0], grad_to_delta_i[1])

            for j in range(self.particle_num_neighbors[frame,i]):
                p_j = self.particle_neighbors[frame, i, j]
                # TODO: does taichi supports break?
                if p_j >= 0:
                    pos_ji = pos_i - self.positions_iter[frame,it,p_j]
                    lambda_j = self.lambdas[frame, it, p_j]

                    grad_j =  self.spiky_gradient_forward(pos_ji, self.h)
                    scorr_ij = self.compute_scorr_forward(pos_ji)

                    # Compute from delta to pos_j through spiky gradient path
                    # delta from each neighbor is computed as 
                    # (lambda_i + lambda_j + scorr_ij) / rho0 * grad_j
                    # To propagate to grad_j, need to get the coefficient
                    y_ji = (lambda_i + lambda_j + scorr_ij) / self.rho0
                    # Each neighbor is used to compute spiky gradient once
                    # and pos_j is negative going into spiky gradient
                    self.positions_iter.grad[frame,it,p_j] += -1 * y_ji * self.spiky_gradient_backward(pos_ji, self.h) @ \
                                                                    grad_to_delta_i
                    # Particle i is used to compute for every neighbor
                    self.positions_iter.grad[frame,it,i] += y_ji * self.spiky_gradient_backward(pos_ji, self.h) @ \
                                                                    grad_to_delta_i

                    # Compute from delta to lambdas
                    # Each neighbor's lambda is added once
                    # print(grad_j[0], grad_j[1])
                    self.lambdas.grad[frame,it,p_j] += 1 / self.rho0 * grad_j.dot(grad_to_delta_i)
                    # Particle i gets contribution for every neighbor
                    self.lambdas.grad[frame,it,i] += 1 / self.rho0 * grad_j.dot(grad_to_delta_i)

                    # Compute from delta to pos_j through scorr path
                    # This is a scalar
                    scorr_to_p_ji = self.compute_scorr_backward(pos_ji)
                    # This is a 2-vector
                    grad_to_scorr = 1 / self.rho0 * grad_j.dot(grad_to_delta_i)
                    # Each neighbor is used to compute scorr once
                    # and pos_j is negative going into scorr
                    self.positions_iter.grad[frame,it,p_j] += - scorr_to_p_ji * grad_to_scorr
                    # Particle i is used to compute for every neighbor
                    self.positions_iter.grad[frame,it,i] += scorr_to_p_ji * grad_to_scorr

    @ti.kernel
    def apply_position_deltas_forward(self, frame: ti.i32, it: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                self.positions_iter[frame, it+1, i] = self.positions_iter[frame,it,i] + self.position_deltas[frame,it,i]
    
    @ti.kernel
    def apply_position_deltas_backward(self, frame: ti.i32, it: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                self.positions_iter.grad[frame,it,i] = self.positions_iter.grad[frame,it+1,i]
                self.position_deltas.grad[frame,it,i] = self.positions_iter.grad[frame,it+1,i]

    @ti.kernel
    def apply_final_position_deltas_forward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos = self.positions_iter[frame,self.pbf_num_iters,i]
                pos_confined = self.confine_position_to_boundary_forward(pos)
                self.positions_after_delta_box[frame, i] = pos_confined
                self.positions_after_delta[frame,i] = self.confine_position_to_box_forward(pos_confined)

                # pos_confined = self.confine_position_to_box_forward(pos)
                # self.positions_after_delta_box[frame, i] = pos_confined
                # self.positions_after_delta[frame,i] = self.confine_position_to_boundary_forward(pos_confined)

    @ti.kernel
    def apply_final_position_deltas_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1 :
                pos = self.positions_iter[frame, self.pbf_num_iters, i]
                jacobian_bounds = self.confine_position_to_boundary_backward(pos)
                pos = self.positions_after_delta_box[frame, i]
                jacobian_box = self.confine_position_to_box_backward(pos)

                self.positions_iter.grad[frame,self.pbf_num_iters,i] = jacobian_box @ jacobian_bounds @ self.positions_after_delta.grad[frame,i]


    @ti.kernel
    def compute_loss_forward(self):
        for f in range(1,self.max_timesteps):
            loss = 0.0
            n = self.num_active[f]
            if n != 0:
                for i in range(self.num_particles):
                    if self.particle_active[f,i] == 1:
                        if self.positions[f,i][1] <= self.boundary[1]-1:
                            # target = ti.Vector([12, 10, 12])
                            d = 0.5 * (self.boundary[1]-1 - self.positions[f,i][1])**2
                            # dif = target - self.positions[f,i]
                            # d = 0.5 * dif.norm()**2
                            if d > 0:
                                loss +=  d 
                self.loss[None] += loss / (self.num_active[f])
        self.loss[None] /= self.max_timesteps


    @ti.kernel
    def compute_loss_backward(self):
        for f in range(1,self.max_timesteps):
            n = self.num_active[f]
            if n != 0:
                for i in range(self.num_particles):
                    if self.particle_active[f,i] == 1:
                        if self.positions[f,i][1] <= self.boundary[1]-1:
                            dif = self.boundary[1]-1 - self.positions[f,i][1]
                            d = 0.5 * (dif)**2
                            # target = ti.Vector([12, 10, 12])
                            # dif = target - self.positions[f,i]
                            # d = 0.5 * dif.norm()**2
                            if d > 0:
                                self.positions.grad[f,i] += ti.Vector([0.0, -1*dif, 0.0]) / self.num_active[f] / self.max_timesteps
                                # self.positions.grad[f,i] += -1 * dif


    def clear_neighbor_info(self):
        self.grid_num_particles.fill(0)
        self.particle_neighbors.fill(-1)
        self.grid2particles.fill(0)
        self.particle_num_neighbors.fill(0)
    
    @ti.kernel
    def copy_active(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.positions[frame-1,i][1] > self.boundary[1] - 1:
                self.particle_active[frame,i] = 2
            else:
                self.particle_active[frame,i] = self.particle_active[frame-1,i]
        self.num_active[frame] = self.num_active[frame-1]
        self.num_suctioned[frame] = self.num_suctioned[frame-1]


    def step_forward(self, frame):
        # Everything active at the end of the previous frame
        # are assumed to be still active
        self.copy_active(frame)

        self.gravity_forward(frame)

        self.update_grid(frame)
        self.find_particle_neighbors(frame)
        for it in range(self.pbf_num_iters):
            self.compute_lambdas_forward(frame,it)
            self.compute_position_deltas_forward(frame,it)
            self.apply_position_deltas_forward(frame,it)
        self.apply_final_position_deltas_forward(frame)

        self.apply_suction_forward(frame)
        self.prop_resting_forward(frame)

        self.update_velocity_froward(frame)

        
    def backward_step(self, frame):
        # print("Frame",frame)
        self.update_velocity_backward(frame)

        self.prop_resting_backward(frame)
        self.apply_suction_backward(frame)

        self.apply_final_position_deltas_backward(frame)

        for it in reversed(range(self.pbf_num_iters)):
            self.clear_local_grads()
            self.apply_position_deltas_backward(frame,it)
            self.compute_position_deltas_backward(frame,it)
            self.compute_lambdas_backward(frame,it)

        self.gravity_backward(frame)

    # For aux simulation that goes through all timesteps
    def forward(self):
        self.clear_neighbor_info()
        # if self.do_emit:
            # self.emit_particles(3, 0, np.array([[1.0, 1.0, 2.0],[1.0, 1.0, 3.0],[1.0, 1.0, 4.0]]), np.array([[10.0, 0.0, 5.0],[10.0, 0.0, 5.0],[10.0, 0.0, 5.0]]))
        if self.do_save_ply:
            self.save_ply(0)
        if self.do_save_npz:
            self.save_npz(0)
        if self.do_save_npy:
            self.save_npy(0)
        for i in range(1,self.max_timesteps):
            # self.move_tool(i)
            self.step_forward(i)
            if self.do_emit:
                # self.emit_particles(3, i, np.array([[1.0, 1.0, 2.0],[1.0, 1.0, 3.0],[1.0, 1.0, 4.0]]), np.array([[10.0, 0.0, 0.0],[10.0, 0.0, 0.0],[10.0, 0.0, 0.0]]))
                self.emit_particles(9, i, np.array([[1.0, 5.0, 0.5],[1.0, 5.0, 1.0],[1.0, 5.0, 1.5],[1.0, 5.0, 2.0],[1.0, 5.0, 2.5],[1.0, 5.0, 3.0],[1.0, 5.0, 3.5],[1.0, 5.0, 4.0],[1.0, 5.0, 4.5]]), 
                                          np.array([[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0]]))
            if self.do_save_ply:
                self.save_ply(i)
            if self.do_save_npz:
                self.save_npz(i)
            if self.do_save_npy:
                self.save_npy(i)
        self.loss[None] = 0
        self.compute_loss_forward()

    # For aux simulation that goes through all timesteps
    def backward(self):
        self.clear_global_grads()
        self.compute_loss_backward()
        for i in reversed(range(1, self.max_timesteps)):
            self.backward_step(i)

    # For actual simulation that takes one step at a time
    def init_step(self):
        self.clear_neighbor_info()
        if self.do_emit:
            # self.emit_particles(3, 0, np.array([[1.0, 1.0, 2.0],[1.0, 1.0, 3.0],[1.0, 1.0, 4.0]]), np.array([[10.0, 0.0, 0.0],[10.0, 0.0, 0.0],[10.0, 0.0, 0.0]]))
            self.emit_particles(9, 0, np.array([[1.0, 5.0, 0.5],[1.0, 5.0, 1.0],[1.0, 5.0, 1.5],[1.0, 5.0, 2.0],[1.0, 5.0, 2.5],[1.0, 5.0, 3.0],[1.0, 5.0, 3.5],[1.0, 5.0, 4.0],[1.0, 5.0, 4.5]]), 
                                          np.array([[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0]]))
        if self.do_save_npy:
            self.save_npy(0)

    def confine_tool_to_boundary(self, pos):
        pos = pos.copy()
        # Global bounds
        if pos[0] <= 0:
            pos[0] = 0
        elif pos[0] + self.tool_dims[None][0] >= self.boundary[0]:
            pos[0] = self.boundary[0] - self.tool_dims[None][0]
        if pos[1] <= 0:
            pos[1] = 0
        elif pos[1] >= 0.5:
            pos[1] = 0.5
        if pos[2] <= 0:
            pos[2] = 0
        elif pos[2] + self.tool_dims[None][2] >= self.boundary[2]:
            pos[2] = self.boundary[2] - self.tool_dims[None][2]
        # Center obstacle
        box_pos = ti.Vector([0,0,10])
        box_dims = ti.Vector([10, 20, 5])

        box_front = box_pos[2]
        box_back = box_pos[2] - box_dims[2]
        box_left = box_pos[0]
        box_right = box_pos[0] + box_dims[0]

        if pos[0] >= box_left and pos[0] <= box_right and pos[2] - self.tool_dims[None][2] <= box_front and pos[2] >= box_back:
            d_right = box_right - pos[0]
            d_front = box_front - pos[2]
            d_back = pos[2] - box_back

            if d_right <= d_front and d_right <= d_back:
                pos[0] = box_right
            elif d_front <= d_right and d_front <= d_back:
                pos[2] = box_front
            else:
                pos[2] = box_back

        return pos

    @ti.kernel
    def move_tool(self, frame: ti.i32, pos: ti.ext_arr()):
        self.tool_states[frame] = ti.Vector([pos[0], pos[1], pos[2]])

    # For actual simulation that takes one step at a time
    def take_action(self, frame, tool_pos):
        self.move_tool(frame, tool_pos)
        self.step_forward(frame)
        if self.do_emit:
            # self.emit_particles(3, frame, np.array([[1.0, 1.0, 2.0],[1.0, 1.0, 3.0],[1.0, 1.0, 4.0]]), np.array([[10.0, 0.0, 0.0],[10.0, 0.0, 0.0],[10.0, 0.0, 0.0]]))
            self.emit_particles(9, frame, np.array([[1.0, 5.0, 0.5],[1.0, 5.0, 1.0],[1.0, 5.0, 1.5],[1.0, 5.0, 2.0],[1.0, 5.0, 2.5],[1.0, 5.0, 3.0],[1.0, 5.0, 3.5],[1.0, 5.0, 4.0],[1.0, 5.0, 4.5]]), 
                                          np.array([[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0]]))
        if self.do_save_npy:
            self.save_npy(frame)

    def save_ply(self,frame):
        ply_writer = ti.PLYWriter(num_vertices=self.num_active[frame])
        pos_np = self.positions.to_numpy()[frame,:,:]
        active_np = self.particle_active.to_numpy()[frame,:]
        save_inds = active_np == 1
        ply_writer.add_vertex_pos(pos_np[save_inds, 0], pos_np[save_inds, 1], pos_np[save_inds, 2])
        series_prefix = "../viz_results/3D/new/ply/frame.ply"
        ply_writer.export_frame_ascii(frame+1, series_prefix)

    def save_npz(self,frame):
        arrs = {}
        arrs["pos"] = self.positions.to_numpy()[frame,:,:]
        arrs["vel"] = self.velocities.to_numpy()[frame,:,:]
        np.savez("../viz_results/3D/new/npz/frame_{}".format(frame) + ".npz", **arrs)

    def save_npy(self,frame):
        pos = self.positions.to_numpy()[frame,:,:]
        active = self.particle_active.to_numpy()[frame,:]
        # inds = np.logical_or(active == 1, active == 2)
        inds = active == 1
        np.save("../viz_results/3D/new_MPC/exp41/particles/frame_{}".format(frame) + ".npy", pos[inds,:])

        tool_pos = self.tool_states.to_numpy()[frame,:]
        np.save("../viz_results/3D/new_MPC/exp41/tool/frame_{}".format(frame) + ".npy", tool_pos)

    