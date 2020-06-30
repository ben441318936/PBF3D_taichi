# 3D Fluid simulation using position-based fluids
# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Based on 2D implementation by Ye Kuang (k-ye)

import taichi as ti
import numpy as np
import math
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

@ti.data_oriented
class DiffFluidSim3D:
    def __init__(self, 
                num_particles=5000, 
                max_timesteps=10,
                boundary=(40,20,40), 
                gui=None, 
                do_render=False, 
                do_render_save=False,
                render_res=(400,400), 
                render_scaling=10, 
                do_print_stats=False, 
                print_frequency=50, 
                do_ply_save=False,
                render_save_dir="./render_frames",
                ply_save_prefix="./ply_frames/frame.ply"):
        
        self.dim = 3
        self.bg_color = 0x112f41
        self.particle_color = 0x068587
        self.boundary_color = 0xebaca2

        self.gui = gui
        self.do_render = do_render
        self.render_res = render_res
        self.render_scaling = render_scaling
        self.do_render_save = do_render_save
        self.render_save_dir = render_save_dir

        self.do_print_stats = do_print_stats
        self.print_frequency = print_frequency
        self.print_counter = 0

        self.color_map = plt.get_cmap("bwr")
        self.do_ply_save = do_ply_save
        self.ply_save_prefix = ply_save_prefix

        self.boundary = np.array(boundary)

        self.cell_size = 2.51
        self.cell_recpr = 1.0 / self.cell_size

        def round_up(f, s):
            return (math.floor(f * self.cell_recpr / s) + 1) * s

        self.grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1))

        self.max_timesteps = max_timesteps
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

        self.target = ti.Vector(self.dim, dt=ti.f32)

        self.total_pos_delta = ti.Vector(self.dim, dt=ti.f32)
        self.positions = ti.Vector(self.dim, dt=ti.f32)
        self.velocities = ti.Vector(self.dim, dt=ti.f32)
        self.particle_active = ti.var(ti.i32)
        self.num_active = ti.var(ti.i32)

        self.num_active_changed = False
        self.particle_rgba = np.zeros((self.num_particles,4))

        # Once taichi supports clear(), we can get rid of grid_num_particles
        self.grid_num_particles = ti.var(ti.i32)
        self.grid2particles = ti.var(ti.i32)
        self.particle_num_neighbors = ti.var(ti.i32)
        self.particle_neighbors = ti.var(ti.i32)
        self.lambdas = ti.var(ti.f32)
        self.lambdas_grad_i = ti.Vector(self.dim, ti.f32)
        self.lambdas_sum_gradient_sqr_i = ti.var(ti.f32)
        self.lambdas_density_constraints_i = ti.var(ti.f32)

        self.position_deltas = ti.Vector(self.dim, dt=ti.f32)
        # 0: x-pos, 1: timestep in sin()
        self.board_states = ti.Vector(2, dt=ti.f32)

        self.loss = ti.var(ti.f32, needs_grad=True)

        self.place_vars()

        print(f'boundary={self.boundary} grid={self.grid_size} cell_size={self.cell_size}')

    def initialize(self):
        self.init_particles()
        self.init_board()

    def place_vars(self):
        ti.root.dense(ti.i, self.num_particles).place(self.total_pos_delta)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions, self.velocities)
        ti.root.dense(ti.i, self.num_particles).place(self.lambdas, self.position_deltas)
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.particle_active)
        ti.root.dense(ti.i, self.max_timesteps).place(self.num_active)

        grid_snode = ti.root.dense(ti.ijk, self.grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l, self.num_particles).place(self.grid2particles)

        nb_node = ti.root.dense(ti.i, self.num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, self.max_num_neighbors).place(self.particle_neighbors)

        ti.root.dense(ti.i, self.num_particles).place(self.lambdas_grad_i)
        ti.root.dense(ti.i, self.num_particles).place(self.lambdas_sum_gradient_sqr_i)
        ti.root.dense(ti.i, self.num_particles).place(self.lambdas_density_constraints_i)

        ti.root.place(self.board_states)
        
        ti.root.place(self.target)
        ti.root.place(self.loss)
        ti.root.lazy_grad()

    @ti.func
    def init_pos_vel(self):
        for i in range(self.num_particles):
            for j in range(self.max_timesteps):
                for c in ti.static(range(self.dim)):
                    self.positions[j,i][c] = 0
                    self.velocities[j,i][c] = 0

    @ti.func
    def init_total_pos_delta(self):
        for i in range(self.num_particles):
            for c in ti.static(range(self.dim)):
                self.total_pos_delta[i][c] = 0

    @ti.func
    def init_particle_active(self):
        for i in range(self.max_timesteps):
            for j in range(self.num_particles):
                self.particle_active[i,j] = 0

    @ti.func
    def init_num_active(self):
        for i in range(self.max_timesteps):
            self.num_active[i] = 0

    @ti.kernel
    def init_particles(self):

        self.init_pos_vel()
        self.init_total_pos_delta()
        self.init_particle_active()
        self.init_num_active()

    # Kernel to emit 1 particle
    @ti.kernel
    def emit_particles_kernel(self, frame: ti.i32, init_pos: ti.ext_arr(), init_vel: ti.ext_arr()):
        for c in ti.static(range(self.dim)):
            # r = 0.0
            # if c != 0:
            #     r = 3.0 * ti.random()
            self.positions[frame, self.num_active[frame]][c] += init_pos[c]
            self.velocities[frame, self.num_active[frame]][c] += init_vel[c]
        #self.num_active[None] += 1

    # Python-scope to control emission and color setting
    def emit_particles(self, num, frame, init_pos, init_vel):
        if frame+1 >= self.max_timesteps:
            return
        if self.num_active[frame] >= self.num_particles:
            return

        color_ind = self.num_active[frame] / self.num_particles
        color_ind = round(0.2 * round(color_ind/0.2) , 1)
        color = self.color_map(color_ind)

        x = int(np.ceil(np.sqrt(num)))
        xy = np.mgrid[-1.2:1.2:x*1j, -1.2:1.2:x*1j].reshape(2,-1).T

        for i in range(num):
            if self.num_active[frame] < self.num_particles:
                offset = np.array([0,xy[i,0],xy[i,1]])
                self.emit_particles_kernel(frame, init_pos + offset, init_vel)
                self.particle_rgba[self.num_active[frame],:] = np.array(color)
                self.particle_active[frame, self.num_active[frame]] = 1
                self.num_active[frame] += 1     


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
        return ti.cast(pos * self.cell_recpr, ti.i32)

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
                p[i] = bmin + self.epsilon
            elif bmax[i] <= p[i]:
                p[i] = bmax[i] - self.epsilon
        return p


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
    def apply_gravity_within_boundary(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                g = ti.Vector([0.0, 0.0, -9.8])
                pos, vel = self.positions[frame-1,i], self.velocities[frame-1,i]
                vel += g * self.time_delta
                pos += vel * self.time_delta
                #self.positions[frame,i] = self.confine_position_to_boundary(pos)
                self.total_pos_delta[i] = self.confine_position_to_boundary(pos) - self.positions[frame-1,i]

    @ti.kernel
    def confine_to_boundary(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos = self.positions[frame-1,i] + self.total_pos_delta[i]
                #self.positions[frame,i] = self.confine_position_to_boundary(pos)
                self.total_pos_delta[i] = self.confine_position_to_boundary(pos) - self.positions[frame-1,i]

    @ti.kernel
    def update_grid(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                cell = self.get_cell(self.positions[frame-1,i] + self.total_pos_delta[i])
                # ti.Vector doesn't seem to support unpacking yet
                # but we can directly use int Vectors as indices
                self.grid2particles[cell, i] = 1


    @ti.kernel
    def find_particle_neighbors(self, frame: ti.i32):
        for i in range(self.num_particles):
            for j in range(self.num_particles):
                if self.particle_active[frame, i] == 1:
                    if i != j and self.particle_num_neighbors[i] < self.max_num_neighbors:
                        pos_i = self.positions[frame-1,i] + self.total_pos_delta[i]
                        cell = self.get_cell(pos_i)
                        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))): 
                            cell_to_check = cell + offs
                            if self.is_in_grid(cell_to_check):
                                if self.grid2particles[cell_to_check, j] == 1:
                                        dist = (pos_i - (self.positions[frame-1,j] + self.total_pos_delta[j])).norm()
                                        if (dist < self.neighbor_radius):
                                            self.particle_neighbors[i, self.particle_num_neighbors[i]] = j
                                            self.particle_num_neighbors[i] += 1

    @ti.kernel
    def compute_lambdas_contris(self, frame: ti.i32):
        # Eq (8) ~ (11)
        for i in range(self.num_particles):
            for j in range(self.particle_num_neighbors[i]):
                if self.particle_active[frame, i] == 1:
                    p_j = self.particle_neighbors[i,j]
                    if p_j >= 0:
                        pos_i = self.positions[frame-1,i] + self.total_pos_delta[i]
                        pos_ji = pos_i - (self.positions[frame-1,p_j] + self.total_pos_delta[p_j])

                        grad_j = self.spiky_gradient(pos_ji, self.h)
                        self.lambdas_grad_i[i] += grad_j
                        self.lambdas_sum_gradient_sqr_i[i] += grad_j.dot(grad_j)
                        # Eq (2)
                        self.lambdas_density_constraints_i[i] += self.poly6_value(pos_ji.norm(), self.h)

    @ti.kernel
    def compute_lambdas(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame, i] == 1:
                # Eq (1)
                density_constraint = (self.mass * self.lambdas_density_constraints_i[i] / self.rho0) - 1.0
                self.lambdas_sum_gradient_sqr_i[i] += self.lambdas_grad_i[i].dot(self.lambdas_grad_i[i])
                self.lambdas[i] = (-density_constraint) / (self.lambdas_density_constraints_i[i] +
                                                        self.lambda_epsilon)

    