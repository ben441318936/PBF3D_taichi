import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

@ti.data_oriented
class HandGradSim:
    def __init__(self):
        self.gui = ti.GUI("PBF2D", res=(400,400))
        self.bg_color = 0x112f41
        self.particle_color = 0x068587
        self.boundary_color = 0xebaca2

        self.render_res = (400,400)
        self.render_scaling = 10

        self.dim = 2
        self.delta_t = 1.0 / 20.0
        self.max_timesteps = 1000
        self.num_particles = 500

        self.boundary = np.array([40.0,40.0])

        self.cell_size = 2.51
        self.cell_recpr = 1.0 / self.cell_size

        def round_up(f, s):
            return (math.floor(f * self.cell_recpr / s) + 1) * s

        self.grid_size = (round_up(self.boundary[0], 1), round_up(self.boundary[1], 1))

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

        self.positions = ti.Vector(self.dim, ti.f32)
        self.positions_after_grav = ti.Vector(self.dim, ti.f32)
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

        self.board_states = ti.Vector(2, dt=ti.f32)
        self.board_i = ti.var(ti.i32)
        # 0: width, 1: height
        self.board_dims = ti.Vector(self.dim, dt=ti.f32)

        self.distance_matrix = ti.var(ti.f32)
        self.min_dist_frame = ti.var(ti.i32)
        self.loss = ti.var(ti.f32)

        self.place_vars()

    def place_vars(self):
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions, self.velocities)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions_after_grav)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.pbf_num_iters+1).dense(ti.k, self.num_particles).place(self.positions_iter)

        ti.root.dense(ti.i, self.num_particles).place(self.distance_matrix)
        ti.root.dense(ti.i, self.num_particles).place(self.min_dist_frame)
        ti.root.place(self.loss)
        ti.root.place(self.target)

        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.particle_active)
        ti.root.dense(ti.i, self.max_timesteps).place(self.num_active)

        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.pbf_num_iters).dense(ti.k, self.num_particles).place(self.position_deltas)

        grid_snode = ti.root.dense(ti.i, self.max_timesteps).dense(ti.jk, self.grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l, self.max_num_particles_per_cell).place(self.grid2particles)

        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.pbf_num_iters).dense(ti.k, self.num_particles).place(self.lambdas)

        nb_node = ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.k, self.max_num_neighbors).place(self.particle_neighbors)

        ti.root.dense(ti.i, self.num_particles).dense(ti.j, self.max_num_neighbors).place(self.SGS_to_grad_j)
        ti.root.dense(ti.i, self.num_particles).dense(ti.j, self.max_num_neighbors).place(self.local_grad_j)

        ti.root.dense(ti.i, self.max_timesteps).place(self.board_states)
        ti.root.place(self.board_i)
        ti.root.place(self.board_dims)

        ti.root.lazy_grad()

    @ti.kernel
    def init_target(self):
        self.target[None] = ti.Vector([11.0, 11.0])

    @ti.kernel
    def init_board(self):
        self.board_states[0] = ti.Vector([20.0, 2.0])
        self.board_i[None] = 0
        self.board_dims[None] = ti.Vector([1.0, 10.0])

    def initialize(self):
        self.positions.fill(0.0)
        self.positions_after_grav.fill(0.0)
        self.positions_iter.fill(0.0)
        self.velocities.fill(0.0)
        self.lambdas.fill(0.0)
        self.position_deltas.fill(0.0)
        self.loss.fill(0.0)
        self.distance_matrix.fill(0.0)
        self.min_dist_frame.fill(0)
        self.particle_active.fill(0)
        self.num_active.fill(0)
        self.clear_neighbor_info()
        # Temporary hardcoded values here
        self.init_target()
        self.init_board()
        

    def clear_global_grads(self):
        self.positions.grad.fill(0.0)
        self.positions_after_grav.grad.fill(0.0)
        self.positions_iter.grad.fill(0.0)
        self.velocities.grad.fill(0.0)
        self.lambdas.grad.fill(0.0)
        self.position_deltas.grad.fill(0.0)
        self.loss.grad.fill(0.0)

    def clear_local_grads(self):
        self.SGS_to_grad_j.fill(0.0)
        self.local_grad_j.fill(0.0)

    @ti.kernel
    def place_particle(self, frame: ti.i32, i: ti.i32, p: ti.ext_arr(), v: ti.ext_arr()):
        pos = ti.Vector([p[0], p[1]])
        pos = self.confine_position_to_boundary_forward(pos)
        for k in ti.static(range(self.dim)):
            self.positions[frame, i][k] = pos[k]
            self.velocities[frame, i][k] = v[k]

    def emit_particles(self, n, frame, p, v):
        for i in range(n):
            if self.num_active[frame] < self.num_particles:
                offset = np.array([0,0])
                self.place_particle(frame, self.num_active[frame], p[i] + offset, v[i])
                self.particle_active[frame, self.num_active[frame]] = 1
                self.num_active[frame] += 1 

    @ti.func
    def confine_position_to_boundary_forward(self, p):
        # Global boundaries
        bmin = self.particle_radius
        bmax = ti.Vector([self.boundary[0], self.boundary[1]]) - self.particle_radius

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
        bmax = ti.Vector([self.boundary[0], self.boundary[1]]) - self.particle_radius

        jacobian = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])

        for i in ti.static(range(self.dim)):
            if p[i] <= bmin or p[i] >= bmax[i]:
                jacobian[i,i] = 0

        return jacobian

    @ti.func
    def confine_position_to_board_forward(self, frame, p0, p1):
        board_left = self.board_states[frame][0]
        board_right = self.board_states[frame][0] + self.board_dims[None][0]
        board_bot = self.board_states[frame][1]
        board_top = self.board_states[frame][1] + self.board_dims[None][1]

        p_proj = ti.Vector([p1[0], p1[1]])

        # If particle is in the interior of the board rect
        if p1[0] >= board_left and p1[0] <= board_right and p1[1] >= board_bot and p1[1] <= board_top:
            # Regular projection, the projection is based on particle's previous location
            if p0[0] <= board_left + 0.1:
                p_proj[0] = board_left - self.epsilon * ti.random()
            elif p0[0] >= board_right - 0.1:
                p_proj[0] = board_right + self.epsilon * ti.random()
            if p0[1] <= board_bot + 0.1:
                p_proj[1] = board_bot - self.epsilon * ti.random()
            elif p0[1] >= board_top - 0.1:
                p_proj[1] = board_top + self.epsilon * ti.random()
            
        return p_proj

    @ti.func
    def confine_position_to_board_backward(self, frame, p0, p1):
        board_left = self.board_states[frame][0]
        board_right = self.board_states[frame][0] + self.board_dims[None][0]
        board_bot = self.board_states[frame][1]
        board_top = self.board_states[frame][1] + self.board_dims[None][1]

        jacobian = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])

        # If particle is in the interior of the board rect
        if p1[0] >= board_left and p1[0] <= board_right and p1[1] >= board_bot and p1[1] <= board_top:
            # Regular projection, the projection is based on particle's previous location
            if p0[0] <= board_left + 0.5:
                jacobian[0,0] = 0
            elif p0[0] >= board_right - 0.5:
                jacobian[0,0] = 0
            if p0[1] <= board_bot + 0.5:
                jacobian[1,1] = 0
            elif p0[1] >= board_top - 0.5:
                jacobian[1,1] = 0

        return jacobian

            
    @ti.kernel
    def gravity_forward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame-1,i] == 1:
                g = ti.Vector([0.0, -9.81])
                pos, vel = self.positions[frame-1,i], self.velocities[frame-1,i]
                vel += g * self.delta_t
                pos += vel * self.delta_t
                self.positions_after_grav[frame,i] = pos
                # The position info going into the solver iterations is the confined version
                positions_after_board = self.confine_position_to_board_forward(frame, self.positions[frame-1,i],pos)
                self.positions_iter[frame,0,i] = self.confine_position_to_boundary_forward(positions_after_board)

    @ti.kernel
    def gravity_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame-1,i] == 1:
                pos = self.positions_after_grav[frame, i]
                pos_confined_to_board = self.confine_position_to_board_forward(frame, self.positions[frame-1,i], pos)
                jacobian_bounds = self.confine_position_to_boundary_backward(pos_confined_to_board)
                jacobian_board = self.confine_position_to_board_backward(frame, self.positions[frame-1,i], pos)
                # positions and velocities contributes to positions_after_grav, so it needs component-wise gating
                self.positions.grad[frame-1,i] += jacobian_board @ jacobian_bounds @ self.positions_iter.grad[frame,0,i]
                self.velocities.grad[frame-1,i] += jacobian_board @ jacobian_bounds @ self.positions_iter.grad[frame,0,i] * self.delta_t

    @ti.kernel
    def apply_suction(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame-1,i]:
                pos_i = self.positions[frame-1,i]

                board_left = self.board_states[frame][0]
                board_right = self.board_states[frame][0] + self.board_dims[None][0]
                board_bot = self.board_states[frame][1]
                board_top = self.board_states[frame][1] + self.board_dims[None][1]

                # If particle is within some area, take it out of simulation
                if pos_i[0] >= board_left and pos_i[0] <= board_right and pos_i[1] >= board_bot - 0.5:
                    self.particle_active[frame-1,i] = 2

    @ti.kernel
    def update_velocity_froward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame-1,i] == 1:
                self.velocities[frame,i] = (self.positions[frame,i] - self.positions[frame-1,i]) / self.delta_t

    @ti.kernel
    def update_velocity_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame-1,i] == 1:
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
        return 0 <= c[0] and c[0] < self.grid_size[0] and 0 <= c[1
            ] and c[1] < self.grid_size[1]

    @ti.kernel
    def update_grid(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame-1,i] == 1:
                cell = self.get_cell(self.positions_iter[frame,0,i])
                # ti.Vector doesn't seem to support unpacking yet
                # but we can directly use int Vectors as indices
                offs = self.grid_num_particles[frame, cell].atomic_add(1)
                self.grid2particles[frame, cell, offs] = i

    @ti.kernel
    def find_particle_neighbors(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame-1,i] == 1:
                pos_i = self.positions_iter[frame,0,i]
                cell = self.get_cell(pos_i)
                nb_i = 0
                for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
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
        result = ti.Vector([0.0, 0.0])
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = self.spiky_grad_factor * x * x
            result = r * g_factor / r_len
        return result

    # Upstream gradient is a 2-vector
    @ti.func
    def spiky_gradient_backward(self, r, h):
        jacobian = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        c = 1
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            f = ti.static(self.spiky_grad_factor)
            jacobian[0,0] =   f / h**4 / r_len \
                            - f / h**4 * r[0]**2 / r_len**3 \
                            - 2 * f / h**5 \
                            + f * r_len \
                            + f * r[0] / r_len  
            jacobian[0,1] = - f / h**4 * r[1] * r[0] / r_len**3 \
                            + f * r[1] * r[0] / r_len
            jacobian[1,0] = jacobian[0,1]
            jacobian[1,1] =   f / h**4 / r_len \
                            - f / h**4 * r[1]**2 / r_len**3 \
                            - 2 * f / h**5 \
                            + f * r_len \
                            + f * r[1] / r_len
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
            if self.particle_active[frame-1,i] == 1:
                pos_i = self.positions_iter[frame,it,i]

                grad_i = ti.Vector([0.0, 0.0])
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
            grad_i = ti.Vector([0.0, 0.0])
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
            SGS_to_pos_i = ti.Vector([0.0, 0.0])
            # There is contribution to pos_i from the poly6 computation of each neighbor
            constraint_to_pos_i = ti.Vector([0.0, 0.0])

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
            if self.particle_active[frame-1,i] == 1:
                pos_i = self.positions_iter[frame,it,i]
                lambda_i = self.lambdas[frame,it,i]

                pos_delta_i = ti.Vector([0.0, 0.0])
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
            if self.particle_active[frame-1,i] == 1:
                self.positions_iter[frame, it+1, i] = self.positions_iter[frame,it,i] + self.position_deltas[frame,it,i]
    
    @ti.kernel
    def apply_position_deltas_backward(self, frame: ti.i32, it: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame-1,i] == 1:
                self.positions_iter.grad[frame,it,i] = self.positions_iter.grad[frame,it+1,i]
                self.position_deltas.grad[frame,it,i] = self.positions_iter.grad[frame,it+1,i]

    @ti.kernel
    def apply_final_position_deltas_forward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame-1,i] == 1:
                pos = self.positions_iter[frame,self.pbf_num_iters,i]
                pos_confined_to_board = self.confine_position_to_board_forward(frame, self.positions[frame-1,i], pos)
                self.positions[frame,i] = self.confine_position_to_boundary_forward(pos_confined_to_board)

    @ti.kernel
    def apply_final_position_deltas_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame-1,i] == 1 :
                pos = self.positions_iter[frame, self.pbf_num_iters, i]
                pos_confined_to_board = self.confine_position_to_board_forward(frame, self.positions[frame-1,i], pos)
                # pos_confined_to_bounds = self.confine_position_to_boundary_forward(pos_confined_to_board)

                jacobian_bounds = self.confine_position_to_boundary_backward(pos_confined_to_board)
                jacobian_board = self.confine_position_to_board_backward(frame, self.positions[frame-1,i], pos)

                self.positions_iter.grad[frame,self.pbf_num_iters,i] = jacobian_board @ jacobian_bounds @ self.positions.grad[frame,i]

    @ti.kernel
    def compute_distances(self):
        for i in range(self.num_particles):
            min_dist = (self.positions[1,i][0] - self.board_states[1][0])**2 + (self.positions[1,i][1] - self.board_states[1][1])**2
            min_dist_frame = 1
            for f in range(0, self.max_timesteps):
                if self.particle_active[f,i] == 2:
                    min_dist = 0
                    min_dist_frame = f
                    break
                elif self.particle_active[f,i] == 1:
                    dist = (self.positions[f,i][0] - self.board_states[f][0])**2 + (self.positions[f,i][1] - self.board_states[f][1])**2
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_frame = f
            self.distance_matrix[i] = min_dist
            self.min_dist_frame[i] = min_dist_frame

    @ti.kernel
    def compute_loss_forward(self):
        # for i in range(self.num_particles):
        #     if self.particle_active[self.max_timesteps-1,i] == 1:
        #         for k in ti.static(range(self.dim)):
        #             self.loss[None] += 1/2 * (self.positions[self.max_timesteps-1,i][k] - self.target[None][k])**2
        for i in range(self.num_particles):
            self.loss[None] += 1/2 * self.distance_matrix[i]


    @ti.kernel
    def compute_loss_backward(self):
        # for i in range(self.num_particles):
        #     if self.particle_active[self.max_timesteps-1,i] == 1:
        #         for k in ti.static(range(self.dim)):
        #             self.positions.grad[self.max_timesteps-1,i][k] += self.positions[self.max_timesteps-1,i][k] - self.target[None][k]
            
        for i in range(self.num_particles):
            f = self.min_dist_frame[i]
            self.positions.grad[f,i] += self.positions[f,i] - self.board_states[f]
            self.board_states.grad[f] += - (self.positions[f,i] - self.board_states[f])

    @ti.kernel
    def move_board(self, frame: ti.i32):
        # probably more accurate to exert force on particles according to hooke's law.
        b = self.board_states[frame-1]
        self.board_i[None] += 1
        period = 200
        vel_strength = 2.0
        if self.board_i >= 2 * period:
            self.board_i = 0
        b[0] += ti.sin(self.board_i[None] * np.pi / period) * vel_strength * self.delta_t
        self.board_states[frame] = b

    def clear_neighbor_info(self):
        self.grid_num_particles.fill(0)
        self.particle_neighbors.fill(-1)
        self.grid2particles.fill(0)
        self.particle_num_neighbors.fill(0)
    
    @ti.kernel
    def copy_active(self, frame: ti.i32):
        for i in range(self.num_particles):
            self.particle_active[frame,i] = self.particle_active[frame-1,i]
            self.num_active[frame] = self.num_active[frame-1]


    def step_forward(self, frame):
        self.gravity_forward(frame)
        self.apply_suction(frame)

        self.update_grid(frame)
        self.find_particle_neighbors(frame)
        for it in range(self.pbf_num_iters):
            self.compute_lambdas_forward(frame,it)
            self.compute_position_deltas_forward(frame,it)
            self.apply_position_deltas_forward(frame,it)
        self.apply_final_position_deltas_forward(frame)

        self.update_velocity_froward(frame)

        self.copy_active(frame)

    def backward_step(self, frame):
        self.update_velocity_backward(frame)

        self.apply_final_position_deltas_backward(frame)
        for it in reversed(range(self.pbf_num_iters)):
            self.clear_local_grads()
            self.apply_position_deltas_backward(frame,it)
            self.compute_position_deltas_backward(frame,it)
            self.compute_lambdas_backward(frame,it)

        self.gravity_backward(frame)

    
    def forward(self):
        self.clear_neighbor_info()
        self.emit_particles(1, 0, np.array([[20.0, 1.0]]), np.array([[10.0, 0.0]]))
        self.render(0)
        for i in range(1,self.max_timesteps):
            self.move_board(i)
            self.step_forward(i)
            self.render(i)
            self.emit_particles(1, i, np.array([[20.0, 1.0]]), np.array([[10.0, 0.0]]))
        self.loss[None] = 0
        self.compute_loss_forward()

    def backward(self):
        self.clear_global_grads()
        self.compute_loss_backward()
        for i in reversed(range(1, self.max_timesteps)):
            self.backward_step(i)

    def render(self,frame):
        canvas = self.gui.canvas
        canvas.clear(self.bg_color)
        pos_np = self.positions.to_numpy()[frame,:,:]

        # Particles
        for i in range(pos_np.shape[0]):
            if self.particle_active[frame,i] == 1:
                for j in range(self.dim):
                    pos_np[i,j] *= self.render_scaling / self.render_res[j]
                self.gui.circle(pos_np[i,[0,1]], radius=self.particle_radius*self.render_scaling, color=self.particle_color)

        # Boundary rect
        canvas.rect(ti.vec(0, 0), 
                    ti.vec(1.0,1.0)
                    ).radius(1.5).color(self.boundary_color).close().finish()

        # Board rect
        botLeftX = self.board_states[frame][0] / self.boundary[0]
        botLeftY = self.board_states[frame][1] / self.boundary[1]
        topRightX = (self.board_states[frame][0]+self.board_dims[None][0]) / self.boundary[0]
        topRightY = (self.board_states[frame][1]+self.board_dims[None][1]) / self.boundary[1]
        canvas.rect(ti.vec(botLeftX, botLeftY), ti.vec(topRightX, topRightY)
                    ).radius(1.5).color(self.boundary_color).close().finish()
        
        self.gui.show()
