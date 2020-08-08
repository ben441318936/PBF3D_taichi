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
        self.rho0 = 1.0
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

        self.target = ti.Vector(self.dim, ti.f32)

        self.particle_active = ti.var(ti.i32)
        self.num_active = ti.var(ti.i32)

        self.num_suctioned = ti.var(ti.i32)

        self.positions = ti.Vector(self.dim, ti.f32)
        self.positions_after_grav = ti.Vector(self.dim, ti.f32)
        self.positions_after_grav_confined = ti.Vector(self.dim, ti.f32)
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

        self.tool_centers = ti.Vector(self.dim, dt=ti.f32)
        self.tool_thetas = ti.var(ti.f32)
        self.tool_dims = ti.Vector(self.dim, dt=ti.f32)
        self.tool_vertices = ti.Vector(self.dim, ti.f32)

        self.board_states = ti.Vector(self.dim, dt=ti.f32)
        self.board_i = ti.var(ti.i32)
        # 0: width, 1: height
        self.board_dims = ti.Vector(self.dim, dt=ti.f32)

        self.distance_matrix = ti.var(ti.f32)
        self.min_dist_frame = ti.var(ti.i32)
        self.loss = ti.var(ti.f32)

        self.particle_age = ti.var(ti.i32)

        self.place_vars()

    def place_vars(self):
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions, self.velocities)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions_after_grav)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions_after_grav_confined)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.pbf_num_iters+1).dense(ti.k, self.num_particles).place(self.positions_iter)

        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.distance_matrix)
        ti.root.dense(ti.i, self.num_particles).place(self.min_dist_frame)
        ti.root.place(self.loss)
        ti.root.place(self.target)

        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.particle_age)

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

        ti.root.dense(ti.i, self.max_timesteps).place(self.board_states)
        ti.root.place(self.board_i)
        ti.root.place(self.board_dims)

        ti.root.dense(ti.i, self.max_timesteps).place(self.tool_centers)
        ti.root.dense(ti.i, self.max_timesteps).place(self.tool_thetas)
        ti.root.place(self.tool_dims)
        ti.root.dense(ti.i, 4).place(self.tool_vertices) # Only for rendering

        ti.root.lazy_grad()

    @ti.kernel
    def init_board(self, states: ti.ext_arr()):
        for i in range(self.max_timesteps):
            for k in ti.static(range(self.dim)):
                self.board_states[i][k] = states[i,k]

    @ti.kernel
    def init_tool_centers(self, states: ti.ext_arr()):
        for i in range(self.max_timesteps):
            for k in ti.static(range(self.dim)):
                self.tool_centers[i][k] = states[i,k]

    @ti.kernel
    def init_board_dim(self, dims: ti.ext_arr()):
        self.board_dims[None] = ti.Vector([dims[0], dims[1], dims[2]])

    @ti.kernel
    def init_tool_dim(self, dims: ti.ext_arr()):
        self.tool_dims[None] = ti.Vector([dims[0], dims[1], dims[2]])

    @ti.kernel
    def init_tool_thetas(self, thetas: ti.ext_arr()):
        for i in range(self.max_timesteps):
            self.tool_thetas[i] = thetas[i]

    def initialize(self, board_states=None, tool_centers=None, tool_thetas=None):
        self.positions.fill(0.0)
        self.positions_after_grav.fill(0.0)
        self.positions_after_grav_confined.fill(0.0)
        self.positions_iter.fill(0.0)
        self.velocities.fill(0.0)
        self.lambdas.fill(0.0)
        self.position_deltas.fill(0.0)
        self.loss.fill(0.0)
        self.distance_matrix.fill(0.0)
        self.min_dist_frame.fill(0)
        self.particle_active.fill(0)
        self.num_active.fill(0)
        self.num_suctioned.fill(0)
        self.clear_neighbor_info()
        # Temporary hardcoded values here
        self.board_states.fill(100.0)
        self.tool_centers.fill(20.0)
        self.tool_thetas.fill(0)
        self.board_i[None] = 0
        self.init_board_dim(np.array([1.0, 5.0, 1.0]))
        self.init_tool_dim(np.array([1.5, 10.0, 1.5]))
        if board_states is not None:
            self.init_board(board_states)
        if tool_centers is not None:
            self.init_tool_centers(tool_centers)
        if tool_thetas is not None:
            self.init_tool_thetas(tool_thetas)
        self.particle_age.fill(0)

        
    def clear_global_grads(self):
        self.positions.grad.fill(0.0)
        self.positions_after_grav.grad.fill(0.0)
        self.positions_after_grav_confined.grad.fill(0.0)
        self.positions_iter.grad.fill(0.0)
        self.velocities.grad.fill(0.0)
        self.lambdas.grad.fill(0.0)
        self.position_deltas.grad.fill(0.0)
        self.board_states.grad.fill(0.0)
        self.tool_centers.grad.fill(0.0)
        self.tool_thetas.grad.fill(0.0)
        self.loss.grad.fill(0.0)

    def clear_local_grads(self):
        self.SGS_to_grad_j.fill(0.0)
        self.local_grad_j.fill(0.0)

    @ti.kernel
    def place_particle(self, frame: ti.i32, i: ti.i32, p: ti.ext_arr(), v: ti.ext_arr()):
        pos = ti.Vector([p[0], p[1], p[2]])
        pos = self.confine_position_to_boundary_forward(pos)
        # pos = self.confine_position_to_board_forward(i, pos, pos)
        for k in ti.static(range(self.dim)):
            self.positions[frame, i][k] = pos[k]
            self.velocities[frame, i][k] = v[k]

    def emit_particles(self, n, frame, p, v, ages=None):
        for i in range(n):
            if self.num_active[frame] < self.num_particles:
                # offset = np.array([0,0,1])
                self.place_particle(frame, self.num_active[frame], p[i], v[i])
                self.particle_active[frame, self.num_active[frame]] = 1
                if ages is not None:
                    self.particle_age[frame, self.num_active[frame]] = ages[i]
                else:
                    self.particle_age[frame, self.num_active[frame]] = 1
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

    @ti.func
    def confine_position_to_boundary_backward(self, p):
        # Global boundaries
        bmin = self.particle_radius
        bmax = ti.Vector([self.boundary[0], self.boundary[1], self.boundary[2]]) - self.particle_radius

        jacobian = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        for i in ti.static(range(self.dim)):
            if p[i] <= bmin or p[i] >= bmax[i]:
                jacobian[i,i] = 0

        return jacobian


    @ti.func
    def confine_position_to_board_forward(self, frame, p1):
        board_left = self.board_states[frame][0]
        board_right = self.board_states[frame][0] + self.board_dims[None][0]
        board_bot = self.board_states[frame][1]
        board_top = self.board_states[frame][1] + self.board_dims[None][1]
        board_front = self.board_states[frame][2]
        board_back = self.board_states[frame][2] - self.board_dims[None][2]

        p_proj = ti.Vector([p1[0], p1[1], p1[2]])

        # If particle is in the interior of the board rect
        if p1[0] >= board_left and p1[0] <= board_right and p1[1] >= board_bot and p1[1] <= board_top and p1[2] <= board_front and p1[2] >= board_back:
            # Regular projection, the projection is based on the closest face to the particle
            d = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            d[0] = ti.abs(p1[0] - board_left)
            d[1] = ti.abs(p1[0] - board_right)
            d[2] = ti.abs(p1[1] - board_bot)
            d[3] = ti.abs(p1[1] - board_top)
            d[4] = ti.abs(p1[2] - board_front)
            d[5] = ti.abs(p1[2] - board_back)

            min_d = d[0]
            ind = 0

            for k in ti.static(range(6)):
                if d[k] < min_d:
                    ind = k

            if ind == 0:
                p_proj[0] = board_left - self.epsilon * ti.random()
            elif ind == 1:
                p_proj[0] = board_right + self.epsilon * ti.random()
            elif ind == 2:
                p_proj[1] = board_bot - self.epsilon * ti.random()
            elif ind == 3:
                p_proj[1] = board_top + self.epsilon * ti.random()
            elif ind == 5:
                p_proj[2] = board_front + self.epsilon * ti.random()
            else:
                p_proj[2] = board_back - self.epsilon * ti.random()

        return p_proj

    @ti.func
    def confine_position_to_board_backward(self, frame, p1):
        board_left = self.board_states[frame][0]
        board_right = self.board_states[frame][0] + self.board_dims[None][0]
        board_bot = self.board_states[frame][1]
        board_top = self.board_states[frame][1] + self.board_dims[None][1]
        board_front = self.board_states[frame][2]
        board_back = self.board_states[frame][2] - self.board_dims[None][2]

        jacobian_p = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # If particle is in the interior of the board rect
        if p1[0] >= board_left and p1[0] <= board_right and p1[1] >= board_bot and p1[1] <= board_top and p1[2] <= board_front and p1[2] >= board_back:
            # Regular projection, the projection is based on the closest face to the particle
            d = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            d[0] = ti.abs(p1[0] - board_left)
            d[1] = ti.abs(p1[0] - board_right)
            d[2] = ti.abs(p1[1] - board_bot)
            d[3] = ti.abs(p1[1] - board_top)
            d[4] = ti.abs(p1[2] - board_front)
            d[5] = ti.abs(p1[2] - board_back)

            min_d = d[0]
            ind = 0

            for k in ti.static(range(6)):
                if d[k] < min_d:
                    ind = k

            if ind == 0:
                jacobian_p[0,0] = 0
            elif ind == 1:
                jacobian_p[0,0] = 0
            elif ind == 2:
                jacobian_p[1,1] = 0
            elif ind == 3:
                jacobian_p[1,1] = 0
            elif ind == 5:
                jacobian_p[2,2] = 0
            else:
                jacobian_p[2,2] = 0

        return jacobian_p

            
    @ti.kernel
    def gravity_forward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                g = ti.Vector([0.0, -9.81, 0.0])
                pos, vel = self.positions[frame-1,i], self.velocities[frame-1,i]
                vel += g * self.delta_t
                pos += vel * self.delta_t
                self.positions_after_grav[frame,i] = pos
                self.positions_after_grav_confined[frame,i] = self.confine_position_to_boundary_forward(pos)

    @ti.kernel
    def gravity_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos = self.positions_after_grav[frame, i]
                jacobian_bounds = self.confine_position_to_boundary_backward(pos)
                self.positions.grad[frame-1,i] += jacobian_bounds @ self.positions_after_grav_confined.grad[frame,i]
                self.velocities.grad[frame-1,i] += jacobian_bounds @ self.positions_after_grav_confined.grad[frame,i] * self.delta_t


    @ti.kernel
    def apply_suction_forward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:   
                pos_i = self.positions_after_grav_confined[frame,i]
                self.positions_iter[frame,0,i] = pos_i 
                board_states = self.board_states[frame]
                dims = self.board_dims[None]

                board_left = board_states[0]
                board_right = board_states[0] + dims[0]
                board_bot = board_states[1]
                board_top = board_states[1] + dims[1]
                board_front = board_states[2]
                board_back = board_states[2] - dims[2]
                board_center = board_states + ti.Vector([1*dims[0]/2, -1, -1*dims[2]/2])

                # If particle is within some area, apply force to it
                # Check for outer bounds
                if pos_i[0] >= board_left-1 and pos_i[0] <= board_right+1 and pos_i[1] >= board_bot-1 and pos_i[1] <= board_bot+1 and pos_i[2] <= board_front+1 and pos_i[2] >= board_back-1:
                    # pos_delta = ti.Vector([0.0, 0.0, 0.0])
                    # Check for innner bounds
                    # This is the field that pulls particles close
                    if ((pos_i[0] < board_left or pos_i[0] > board_right) or (pos_i[2] > board_front or pos_i[2] < board_back)) or (pos_i[1] < board_bot):
                        tmp = board_center - pos_i
                        unit = tmp / tmp.norm()
                        # pos_delta += unit * 1 / (tmp.norm()**2 + 2) # Max is 0.5x of the unit vector
                        self.positions_iter[frame,0,i] += unit * 4 / (tmp.norm()**2 + 2)
                    # This is the field that pulls particles vertically up when they are inside
                    if pos_i[0] > board_left and pos_i[0] < board_right and pos_i[2] < board_front and pos_i[2] > board_back and pos_i[1] > board_bot-0.5:
                        # pos_delta += ti.Vector([0.0, 1.0, 0.0]) * 20
                        # No longer interact with other particles
                        self.particle_active[frame,i] = 2
                        self.num_suctioned[frame] += 1
                        self.positions_iter[frame,0,i] += ti.Vector([0.0, 1.0, 0.0]) * 5
                    # self.positions_iter[frame,0,i] = pos_i + pos_delta
            elif self.particle_active[frame,i] == 2:   
                self.positions_iter[frame,0,i] = self.positions[frame-1,i] + ti.Vector([0.0, 1.0, 0.0]) * 5

    # This backward will propagate back to particle position and to tool position
    @ti.kernel
    def apply_suction_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos_i = self.positions_after_grav_confined[frame,i]

                board_states = self.board_states[frame]
                dims = self.board_dims[None]

                board_left = board_states[0]
                board_right = board_states[0] + dims[0]
                board_bot = board_states[1]
                board_top = board_states[1] + dims[1]
                board_front = board_states[2]
                board_back = board_states[2] - dims[2]
                board_center = board_states + ti.Vector([1*dims[0]/2, -0.5, -1*dims[2]/2])

                upstream_grad = self.positions_iter.grad[frame,0,i]

                eye = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                grad_delta = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

                if pos_i[0] >= board_left-1 and pos_i[0] <= board_right+1 and pos_i[1] >= board_bot-1 and pos_i[1] <= board_bot+1 and pos_i[2] <= board_front+1 and pos_i[2] >= board_back-1:
                    # Check for innner bounds
                    # This is the field that pulls particles close
                    if ((pos_i[0] < board_left or pos_i[0] > board_right) or (pos_i[2] > board_front or pos_i[2] < board_back)) or (pos_i[1] < board_bot):
                        tmp = board_center - pos_i
                        norm = tmp.norm()

                        c = 2

                        # d Dx / dx
                        grad_delta[0,0] = norm**(-1) * (norm**2+c)**(-1) \
                                          - tmp[0]**2 * norm**(-3) * (norm**2+c)**(-1) \
                                          - 2 * tmp[0]**2 * norm**(-1) * (norm**2+c)**(-2)
                        # d Dx / dy
                        grad_delta[1,0] = -1 * tmp[0] * tmp[1] * norm**(-3) * (norm**2+c)**(-1) \
                                          - 2 * tmp[0] * tmp[1] * norm**(-1) * (norm**2+c)**(-2)
                        # d Dx / dz
                        grad_delta[2,0] = -1 * tmp[0] * tmp[2] * norm**(-3) * (norm**2+c)**(-1) \
                                          - 2 * tmp[0] * tmp[2] * norm**(-1) * (norm**2+c)**(-2)       

                        # d Dy / dx
                        grad_delta[0,1] = -1 * tmp[1] * tmp[0] * norm**(-3) * (norm**2+c)**(-1) \
                                          - 2 * tmp[1] * tmp[0] * norm**(-1) * (norm**2+c)**(-2)
                        # d Dy / dy
                        grad_delta[1,1] = norm**(-1) * (norm**2+c)**(-1) \
                                          - tmp[1]**2 * norm**(-3) * (norm**2+c)**(-1) \
                                          - 2 * tmp[1]**2 * norm**(-1) * (norm**2+c)**(-2)
                        # d Dy / dz
                        grad_delta[2,0] = -1 * tmp[1] * tmp[2] * norm**(-3) * (norm**2+c)**(-1) \
                                          - 2 * tmp[1] * tmp[2] * norm**(-1) * (norm**2+c)**(-2)   
                        # d Dz / dx
                        grad_delta[0,2] = -1 * tmp[2] * tmp[0] * norm**(-3) * (norm**2+c)**(-1) \
                                          - 2 * tmp[2] * tmp[0] * norm**(-1) * (norm**2+c)**(-2)
                        # d Dz / dy
                        grad_delta[1,2] = -1 * tmp[2] * tmp[1] * norm**(-3) * (norm**2+c)**(-1) \
                                          - 2 * tmp[2] * tmp[1] * norm**(-1) * (norm**2+c)**(-2)
                        # d Dy / dy
                        grad_delta[2,2] = norm**(-1) * (norm**2+c)**(-1) \
                                          - tmp[2]**2 * norm**(-3) * (norm**2+c)**(-1) \
                                          - 2 * tmp[2]**2 * norm**(-1) * (norm**2+c)**(-2)    

                self.positions_after_grav_confined.grad[frame,i] += 4 * -1 * (eye + grad_delta) @ upstream_grad
                self.board_states.grad[frame] += 4 * 1 * grad_delta @ upstream_grad

            elif self.particle_active[frame,i] == 2:
                self.positions.grad[frame,i] = self.positions_iter.grad[frame,0,i]

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
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            f = ti.static(self.spiky_grad_factor)
            # d Gx / dx
            jacobian[0,0] =   f / h**4 / r_len \
                            - f / h**4 * r[0]**2 / r_len**3 \
                            - 2 * f / h**5 \
                            + f * r_len \
                            + f * r[0] / r_len  
            # d Gx / dy
            jacobian[1,0] = - f / h**4 * r[0] * r[1] / r_len**3 \
                            + f * r[0] * r[1] / r_len
            # d Gx / dz
            jacobian[2,0] = - f / h**4 * r[0] * r[2] / r_len**3 \
                            + f * r[0] * r[2] / r_len
            # d Gy / dx
            jacobian[0,1] = - f / h**4 * r[1] * r[0] / r_len**3 \
                            + f * r[1] * r[0] / r_len
            # d Gy / dy
            jacobian[1,1] =   f / h**4 / r_len \
                            - f / h**4 * r[1]**2 / r_len**3 \
                            - 2 * f / h**5 \
                            + f * r_len \
                            + f * r[1] / r_len
            # d Gy / dz
            jacobian[2,1] = - f / h**4 * r[1] * r[2] / r_len**3 \
                            + f * r[1] * r[2] / r_len
            # d Gz / dx
            jacobian[0,2] = - f / h**4 * r[2] * r[0] / r_len**3 \
                            + f * r[2] * r[0] / r_len
            # d Gz / dy
            jacobian[1,2] = - f / h**4 * r[2] * r[1] / r_len**3 \
                            + f * r[2] * r[1] / r_len   
            # d Gy / dy
            jacobian[2,2] =   f / h**4 / r_len \
                            - f / h**4 * r[2]**2 / r_len**3 \
                            - 2 * f / h**5 \
                            + f * r_len \
                            + f * r[2] / r_len          
        for i in ti.static(range(self.dim)):
            for j in ti.static(range(self.dim)):
                if jacobian[i,j] > c:
                    jacobian[i,j] = c
                if jacobian[i,j] < -c:
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
        if 0 < s_sqr and s_sqr < h**2:
            out_grad = self.poly6_factor / h**9 * 3 * (h**2 - s_sqr) #* ti.Vector([1.0, 1.0])
            c = 1
            if out_grad >= c:
                out_grad = c
            elif out_grad <= -c:
                out_grad = -c
            return out_grad
        else:
            return 0.0

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

            c = 1

            if grad_to_lambda_i > c:
                grad_to_lambda_i = c
            elif grad_to_lambda_i < -c:
                grad_to_lambda_i = -c
            # print("Frame", frame, "particle", i, grad_to_lambda_i)

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

            lambda_to_SGS = density_constraint / (sum_gradient_sqr + self.epsilon)**2
            lambda_to_constraint = -1 / (sum_gradient_sqr + self.epsilon)

            # There is contribution to pos_i from the spiky computation of each neighbor
            SGS_to_pos_i = ti.Vector([0.0, 0.0, 0.0])
            # There is contribution to pos_i from the poly6 computation of each neighbor
            constraint_to_pos_i = ti.Vector([0.0, 0.0, 0.0])

            constraint_to_val = self.rho0 / self.mass

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
                # pos_confined_to_board = self.confine_position_to_board_forward(frame, pos)
                self.positions[frame,i] = self.confine_position_to_boundary_forward(pos)

    @ti.kernel
    def apply_final_position_deltas_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1 :
                pos = self.positions_iter[frame, self.pbf_num_iters, i]
                # pos_confined_to_board = self.confine_position_to_board_forward(frame, pos)
                # pos_confined_to_bounds = self.confine_position_to_boundary_forward(pos_confined_to_board)

                jacobian_bounds = self.confine_position_to_boundary_backward(pos)
                # jacobian_board = self.confine_position_to_board_backward(frame, pos)

                self.positions_iter.grad[frame,self.pbf_num_iters,i] = jacobian_bounds @ self.positions.grad[frame,i]


    @ti.kernel
    def compute_loss_forward(self):
        for f in range(1,self.max_timesteps):
            loss = 0.0
            n = self.num_active[f]
            if n != 0:
                for i in range(self.num_particles):
                    if self.particle_active[f,i] == 1 or self.particle_active[f,i] == 2:
                        if self.positions[f,i][1] <= 100:
                            d = 0.5 * (100 - self.positions[f,i][1])**2
                            if d > 0:
                                loss +=  (self.max_timesteps / ti.cast(f, ti.f32)) * d
                self.loss[None] += loss / (self.num_active[f])
        self.loss[None] /= self.max_timesteps


    @ti.kernel
    def compute_loss_backward(self):
        for f in range(1,self.max_timesteps):
            n = self.num_active[f]
            if n != 0:
                for i in range(self.num_particles):
                    if self.particle_active[f,i] == 1 or self.particle_active[f,i] == 2:
                        if self.positions[f,i][1] <= 100:
                            dif = 100 - self.positions[f,i][1]
                            d = 0.5 * (dif)**2
                            if d > 0:
                                self.positions.grad[f,i] += ti.Vector([0.0, -1*dif, 0.0]) / self.num_active[f] / self.max_timesteps


    def clear_neighbor_info(self):
        self.grid_num_particles.fill(0)
        self.particle_neighbors.fill(-1)
        self.grid2particles.fill(0)
        self.particle_num_neighbors.fill(0)
    
    @ti.kernel
    def copy_active(self, frame: ti.i32):
        for i in range(self.num_particles):
            self.particle_active[frame,i] = self.particle_active[frame-1,i]
            self.particle_age[frame,i] = self.particle_age[frame-1,i]
        self.num_active[frame] = self.num_active[frame-1]
        self.num_suctioned[frame] = self.num_suctioned[frame-1]


    def step_forward(self, frame):
        # Everything active at the end of the previous frame
        # are assumed to be still active
        self.copy_active(frame)

        self.gravity_forward(frame)
        # Take move particles with suction
        self.apply_suction_forward(frame)
        self.prop_resting_forward(frame)

        self.update_grid(frame)
        self.find_particle_neighbors(frame)
        for it in range(self.pbf_num_iters):
            self.compute_lambdas_forward(frame,it)
            self.compute_position_deltas_forward(frame,it)
            self.apply_position_deltas_forward(frame,it)
        self.apply_final_position_deltas_forward(frame)

        self.update_velocity_froward(frame)

        
    def backward_step(self, frame):
        self.update_velocity_backward(frame)

        self.apply_final_position_deltas_backward(frame)
        for it in reversed(range(self.pbf_num_iters)):
            self.clear_local_grads()
            self.apply_position_deltas_backward(frame,it)
            self.compute_position_deltas_backward(frame,it)
            self.compute_lambdas_backward(frame,it)

        self.prop_resting_backward(frame)
        self.apply_suction_backward(frame)
        self.gravity_backward(frame)

    # For aux simulation that goes through all timesteps
    def forward(self):
        self.clear_neighbor_info()
        if self.do_emit:
            self.emit_particles(3, 0, np.array([[10.0, 1.0, 10.0],[10.0, 1.0, 11.0],[10.0, 1.0, 9.0]]), np.array([[10.0, 0.0, 5.0],[10.0, 0.0, 5.0],[10.0, 0.0, 5.0]]))
        if self.do_save_ply:
            self.save_ply(0)
        if self.do_save_npz:
            self.save_npz(0)
        if self.do_save_npy:
            self.save_npy(0)
        for i in range(1,self.max_timesteps):
            # self.move_board(i)
            self.step_forward(i)
            if self.do_emit:
                self.emit_particles(3, i, np.array([[10.0, 1.0, 10.0],[10.0, 1.0, 11.0],[10.0, 1.0, 9.0]]), np.array([[10.0, 0.0, 5.0],[10.0, 0.0, 5.0],[10.0, 0.0, 5.0]]))
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
            self.emit_particles(3, 0, np.array([[10.0, 1.0, 10.0],[10.0, 1.0, 11.0],[10.0, 1.0, 9.0]]), np.array([[10.0, 0.0, 5.0],[10.0, 0.0, 5.0],[10.0, 0.0, 5.0]]))
        if self.do_save_npy:
            self.save_npy(0)

    def confine_board_to_boundary(self, pos):
        pos = pos.copy()
        if pos[0] <= 0:
            pos[0] = 0
        elif pos[0] + self.board_dims[None][0] >= self.boundary[0]:
            pos[0] = self.boundary[0] - self.board_dims[None][0]
        if pos[1] <= 0:
            pos[1] = 0
        if pos[2] <= 0:
            pos[2] = 0
        elif pos[2] + self.board_dims[None][2] >= self.boundary[2]:
            pos[2] = self.boundary[2] - self.board_dims[None][2]

        return pos

    @ti.kernel
    def move_board(self, frame: ti.i32, pos: ti.ext_arr()):
        self.board_states[frame] = ti.Vector([pos[0], pos[1], pos[2]])

    # For actual simulation that takes one step at a time
    def take_action(self, frame, tool_pos):
        self.move_board(frame, tool_pos)
        self.step_forward(frame)
        if self.do_emit:
            self.emit_particles(3, frame, np.array([[10.0, 1.0, 10.0],[10.0, 1.0, 11.0],[10.0, 1.0, 9.0]]), np.array([[10.0, 0.0, 5.0],[10.0, 0.0, 5.0],[10.0, 0.0, 5.0]]))
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
        inds = np.logical_or(active == 1, active == 2)
        np.save("../viz_results/3D/new_MPC/exp7/particles/frame_{}".format(frame) + ".npy", pos[inds,:])

        tool_pos = self.board_states.to_numpy()[frame,:]
        np.save("../viz_results/3D/new_MPC/exp7/tool/frame_{}".format(frame) + ".npy", tool_pos)

    
