# 3D Fluid simulation using position-based fluids
# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Based on 2D implementation by Ye Kuang (k-ye)

import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

@ti.data_oriented
class FluidSim3D:
    def __init__(self, num_particles=5000, boundary=(40,20,40), gui=None, do_render=True, render_res=(400,400), render_scaling=10, do_print_stats=True, print_frequency=50):
        self.dim = 3
        self.bg_color = 0x112f41
        self.particle_color = 0x068587
        self.boundary_color = 0xebaca2

        self.gui = gui
        self.do_render = do_render
        self.render_res = render_res
        self.render_scaling = render_scaling

        self.do_print_stats = do_print_stats
        self.print_frequency = print_frequency
        self.print_counter = 0

        self.boundary = np.array(boundary)

        self.cell_size = 2.51
        self.cell_recpr = 1.0 / self.cell_size

        def round_up(f, s):
            return (math.floor(f * self.cell_recpr / s) + 1) * s

        self.grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1))

        self.num_particles = num_particles
        self.max_num_particles_per_cell = 200
        self.max_num_neighbors = 200
        self.time_delta = 1.0 / 20.0
        self.epsilon = 1e-5
        self.particle_radius_in_world = 0.3
        self.particle_radius = self.particle_radius_in_world * render_scaling

        # PBF params
        self.h = 1.1
        self.mass = 1.0
        self.rho0 = 1.0
        self.lambda_epsilon = 100.0
        self.vorticity_epsilon = 0.01
        self.viscosity_c = 0.1
        self.pbf_num_iters = 5
        self.corr_deltaQ_coeff = 0.3
        self.corrK = 0.001
        # Need ti.pow()
        # corrN = 4.0
        self.neighbor_radius = self.h * 1.05

        self.poly6_factor = 315.0 / 64.0 / np.pi
        self.spiky_grad_factor = -45.0 / np.pi

        self.old_positions = ti.Vector(self.dim, dt=ti.f32)
        self.positions = ti.Vector(self.dim, dt=ti.f32)
        self.velocities = ti.Vector(self.dim, dt=ti.f32)
        self.particle_active = ti.var(ti.i32)
        self.num_active = ti.var(ti.i32)
        # Once taichi supports clear(), we can get rid of grid_num_particles
        self.grid_num_particles = ti.var(ti.i32)
        self.grid2particles = ti.var(ti.i32)
        self.particle_num_neighbors = ti.var(ti.i32)
        self.particle_neighbors = ti.var(ti.i32)
        self.lambdas = ti.var(ti.f32)
        self.position_deltas = ti.Vector(self.dim, dt=ti.f32)
        # 0: x-pos, 1: timestep in sin()
        self.board_states = ti.Vector(2, dt=ti.f32)

    def initialize(self):
        self.place_vars()
        self.init_particles
        self.init_board()
        print(f'boundary={self.boundary} grid={self.grid_size} cell_size={self.cell_size}')

    def place_vars(self):
        ti.root.dense(ti.i, self.num_particles).place(self.old_positions, self.positions, self.velocities)
        grid_snode = ti.root.dense(ti.ijk, self.grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l, self.max_num_particles_per_cell).place(self.grid2particles)
        nb_node = ti.root.dense(ti.i, self.num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, self.max_num_neighbors).place(self.particle_neighbors)
        ti.root.dense(ti.i, self.num_particles).place(self.lambdas, self.position_deltas)
        ti.root.place(self.board_states)
        ti.root.dense(ti.i, self.num_particles).place(self.particle_active)
        ti.root.place(self.num_active)

    @ti.func
    def init(self, p: ti.ext_arr(), v: ti.ext_arr()):
        for i in range(self.num_particles):
            for c in ti.static(range(self.dim)):
                self.positions[i][c] = p[i, c]
                self.velocities[i][c] = v[i, c]

    @ti.kernel
    def init_particles(self):
        # np_positions = np.random.uniform([self.particle_radius_in_world, self.boundary[1]/2, self.particle_radius_in_world], 
        #                                 np.array([self.board_states[None][0]/5, self.boundary[1], self.boundary[2]/5]) - self.particle_radius_in_world,
        #                                 (self.num_particles,self.dim))
        # np_velocities = (np.random.rand(self.num_particles, self.dim).astype(np.float32) -
        #                 0.5) * 4.0
        np_positions = 10 * (np.ones((self.num_particles, self.dim)) + ti.random())
        np_velocities = np.zeros((self.num_particles, self.dim))

        self.init(np_positions, np_velocities)
        self.particle_active.fill(0)
        self.num_active[None] = 0


    @ti.kernel
    def emit_particles(self, num: ti.i32, init_pos: ti.ext_arr(), init_vel: ti.ext_arr()):
        if self.num_active[None] < self.num_particles:
            for i in range(num):
                if self.num_active[None] < self.num_particles:
                    self.particle_active[self.num_active[None]] = 1
                    for c in ti.static(range(self.dim)):
                        self.positions[self.num_active[None]][c] = init_pos[c] + ti.random()
                        self.velocities[self.num_active[None]][c] = init_vel[c]
                    self.num_active[None] += 1

    @ti.kernel
    def init_board(self):
        self.board_states[None] = ti.Vector([self.boundary[0] - self.epsilon, -0.0])
    
    @ti.func
    def poly6_value(self, s, h):
        result = 0.0
        if 0 < s and s < h:
            x = (h * h - s * s) / (h * h * h)
            result = self.poly6_factor * x * x * x
        return result

    @ti.func
    def spiky_gradient(self, r, h):
        result = ti.Vector([0.0, 0.0, 0.0])
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = self.spiky_grad_factor * x * x
            result = r * g_factor / r_len
        return result

    @ti.func
    def compute_scorr(self, pos_ji):
        # Eq (13)
        x = self.poly6_value(pos_ji.norm(), self.h) / self.poly6_value(self.corr_deltaQ_coeff * self.h, self.h)
        # pow(x, 4)
        x = x * x
        x = x * x
        return (-self.corrK) * x

    @ti.func
    def get_cell(self, pos):
        return (pos * self.cell_recpr).cast(int)

    @ti.func
    def is_in_grid(self,c):
        # @c: Vector(i32)
        return 0 <= c[0] and c[0] < self.grid_size[0] and 0 <= c[1
            ] and c[1] < self.grid_size[1] and 0 <= c[2] and c[2] < self.grid_size[2]

    @ti.func
    def confine_position_to_boundary(self,p):
        bmin = self.particle_radius_in_world
        # First coordinate is for the x position of the board, which only moves in x
        bmax = ti.Vector([self.board_states[None][0], self.boundary[1], self.boundary[2]]) - self.particle_radius_in_world
        for i in ti.static(range(self.dim)):
            # Use randomness to prevent particles from sticking into each other after clamping
            if p[i] <= bmin:
                p[i] = bmin + self.epsilon * ti.random()
            elif bmax[i] <= p[i]:
                p[i] = bmax[i] - self.epsilon * ti.random()
        return p

    @ti.kernel
    def blit_buffers(self, f: ti.template(), t: ti.template()):
        for i in f:
            t[i] = f[i]

    @ti.kernel
    def move_board(self):
        # probably more accurate to exert force on particles according to hooke's law.
        b = self.board_states[None]
        b[1] += 1.0
        period = 90
        vel_strength = 8.0
        if b[1] >= 2 * period:
            b[1] = 0
        b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * self.time_delta
        self.board_states[None] = b

    @ti.kernel
    def apply_gravity_within_boundary(self):
        for i in self.positions:
            if self.particle_active[i] == 1:
                g = ti.Vector([0.0, 0.0, -9.8])
                pos, vel = self.positions[i], self.velocities[i]
                vel += g * self.time_delta
                pos += vel * self.time_delta
                self.positions[i] = self.confine_position_to_boundary(pos)

    @ti.kernel
    def confine_to_boundary(self):
        for i in self.positions:
            if self.particle_active[i] == 1:
                pos = self.positions[i]
                self.positions[i] = self.confine_position_to_boundary(pos)

    @ti.kernel
    def update_grid(self):
        for i in self.positions:
            if self.particle_active[i] == 1:
                cell = self.get_cell(self.positions[i])
                # ti.Vector doesn't seem to support unpacking yet
                # but we can directly use int Vectors as indices
                offs = self.grid_num_particles[cell].atomic_add(1)
                self.grid2particles[cell, offs] = i

    @ti.kernel
    def find_particle_neighbors(self):
        for i in self.positions:
            if self.particle_active[i] == 1:
                pos_i = self.positions[i]
                cell = self.get_cell(pos_i)
                nb_i = 0
                for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                    cell_to_check = cell + offs
                    if self.is_in_grid(cell_to_check):
                        for j in range(self.grid_num_particles[cell_to_check]):
                            p_j = self.grid2particles[cell_to_check, j]
                            if nb_i < self.max_num_neighbors and p_j != i and (pos_i - self.positions[p_j]).norm() < self.neighbor_radius:
                                self.particle_neighbors[i, nb_i] = p_j
                                nb_i += 1
                self.particle_num_neighbors[i] = nb_i

    @ti.kernel
    def compute_lambdas(self):
        # Eq (8) ~ (11)
        for i in self.positions:
            if self.particle_active[i] == 1:
                pos_i = self.positions[i]

                grad_i = ti.Vector([0.0, 0.0, 0.0])
                sum_gradient_sqr = 0.0
                density_constraint = 0.0

                for j in range(self.particle_num_neighbors[i]):
                    p_j = self.particle_neighbors[i, j]
                    # TODO: does taichi supports break?
                    if p_j >= 0:
                        pos_ji = pos_i - self.positions[p_j]
                        # grad_j is gradient for jth neighbor
                        # with respect to p_i
                        grad_j = self.spiky_gradient(pos_ji, self.h)
                        grad_i += grad_j
                        sum_gradient_sqr += grad_j.dot(grad_j)
                        # Eq(2)
                        density_constraint += self.poly6_value(pos_ji.norm(), self.h)

                # Eq(1)
                density_constraint = (self.mass * density_constraint / self.rho0) - 1.0

                sum_gradient_sqr += grad_i.dot(grad_i)
                self.lambdas[i] = (-density_constraint) / (sum_gradient_sqr +
                                                        self.lambda_epsilon)


    @ti.kernel
    def compute_position_deltas(self):
        # Eq(12), (14)
        for i in self.positions:
            if self.particle_active[i] == 1:
                pos_i = self.positions[i]
                lambda_i = self.lambdas[i]

                pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
                for j in range(self.particle_num_neighbors[i]):
                    p_j = self.particle_neighbors[i, j]
                    # TODO: does taichi supports break?
                    if p_j >= 0:
                        lambda_j = self.lambdas[p_j]
                        pos_ji = pos_i - self.positions[p_j]
                        scorr_ij = self.compute_scorr(pos_ji)
                        pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                            self.spiky_gradient(pos_ji, self.h)

                pos_delta_i /= self.rho0
                self.position_deltas[i] = pos_delta_i

    @ti.kernel
    def apply_position_deltas(self):
        for i in self.positions:
            if self.particle_active[i] == 1:
                self.positions[i] += self.position_deltas[i]

    @ti.kernel
    def update_velocities(self):
        for i in self.positions:
            if self.particle_active[i] == 1:
                self.velocities[i] = (self.positions[i] - self.old_positions[i]) / self.time_delta

    # This is currently broken, gives infs
    @ti.kernel
    def vorticity_confinement(self):
        # Eq (15) ~ (16)
        for i in self.positions:
            if self.particle_active[i] == 1:
                pos_i = self.positions[i]
                vel_i = self.velocities[i]
                omega_i = ti.Vector([0.0, 0.0, 0.0])
                # This is eta in the paper, which is the gradient of the magnitude of omega
                omega_mag_i_grad = ti.Vector([0.0, 0.0, 0.0])

                for j in range(self.particle_num_neighbors[i]):
                    p_j = self.particle_neighbors[i, j]
                    if p_j >= 0:
                        pos_ji = pos_i - self.positions[p_j]
                        vel_ij = self.velocities[p_j] - vel_i
                        # We need gradient with respect to p_j
                        # This is the negative of the normal version
                        # Since we have p_i - p_j
                        grad_j = - self.spiky_gradient(pos_ji, self.h)
                        z = vel_ij.cross(grad_j)
                        # Hand-derived gradient for the cross product
                        z_grad = ti.Matrix([[0.0, vel_ij[2], -vel_ij[1]], [-vel_ij[2], 0.0, vel_ij[0]], [vel_ij[1], -vel_ij[0], 0.0]])
                        omega_mag_i_grad += z_grad @ z
                        omega_i += vel_ij.cross(z)
                omega_mag_i = omega_i.norm()
                # This normalization comes from the gradient itself
                omega_mag_i_grad /= omega_mag_i

                # This normalization comes from Eq (16)
                location_vector = omega_mag_i_grad.normalized()
                f_vorticity = self.vorticity_epsilon * (location_vector.cross(omega_i))
                self.velocities[i] += self.time_delta * f_vorticity

    @ti.kernel
    def apply_XSPH_viscosity(self):
        # Eq (17)
        for i in self.velocities:
            if self.particle_active[i] == 1:
                pos_i = self.positions[i]
                vel_i = self.velocities[i]
                v_delta_i = ti.Vector([0.0, 0.0, 0.0])

                for j in range(self.particle_num_neighbors[i]):
                    p_j = self.particle_neighbors[i, j]
                    pos_ji = pos_i - self.positions[p_j]
                    vel_ij = self.velocities[p_j] - vel_i
                    if p_j >= 0:
                        pos_ji = pos_i - self.positions[p_j]
                        v_delta_i += vel_ij * self.poly6_value(pos_ji.norm(), self.h)

                self.velocities[i] += self.viscosity_c * v_delta_i

    def run_pbf(self):
        self.blit_buffers(self.positions, self.old_positions)
        self.apply_gravity_within_boundary()

        self.grid_num_particles.fill(0)
        self.particle_neighbors.fill(-1)
        self.update_grid()
        self.find_particle_neighbors()
        for _ in range(self.pbf_num_iters):
            self.compute_lambdas()
            self.compute_position_deltas()
            self.apply_position_deltas()

        self.confine_to_boundary()
        self.update_velocities()
        #vorticity_confinement()
        self.apply_XSPH_viscosity()

    def render(self,frame):
        result_dir = "./viz_results/x_z_emit/frames"
        if self.gui is None and self.gui.running:
            print("Can't render without running GUI.")
            return
        canvas = self.gui.canvas
        canvas.clear(self.bg_color)
        pos_np = self.positions.to_numpy()
        active_np = self.particle_active.to_numpy()
        for i in range(pos_np.shape[0]):
            if active_np[i] == 1:
                for j in range(self.dim):
                    if j == 2:
                        pos_np[i,j] *= self.render_scaling / self.render_res[1]
                    else:
                        pos_np[i,j] *= self.render_scaling / self.render_res[j]
                self.gui.circle(pos_np[i,[0,2]], radius=self.particle_radius, color=self.particle_color)
        canvas.rect(ti.vec(0, 0), 
                    ti.vec(self.board_states[None][0] / self.boundary[0],1.0)
                    ).radius(1.5).color(self.boundary_color).close().finish()
        self.gui.show("{}/{:03d}.png".format(result_dir,frame))

    def print_stats(self):
        print('PBF stats:')
        num = self.grid_num_particles.to_numpy()
        avg, max = np.mean(num), np.max(num)
        print(f'  #particles per cell: avg={avg:.2f} max={max}')
        num = self.particle_num_neighbors.to_numpy()
        avg, max = np.mean(num), np.max(num)
        print(f'  #neighbors per particle: avg={avg:.2f} max={max}')

    def step(self,frame):
        self.move_board()
        self.run_pbf()
        if self.do_render:
            self.render(frame)
        if self.do_print_stats:
            self.print_counter += 1
            if self.print_counter == self.print_frequency:
                self.print_stats()
                self.print_counter = 0
            


    