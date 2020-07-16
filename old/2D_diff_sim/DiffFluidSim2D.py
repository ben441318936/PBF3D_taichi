# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Based on 2D implementation by Ye Kuang (k-ye)

import taichi as ti
import numpy as np
import math
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)


@ti.data_oriented
class DiffFluidSim2D:
    def __init__(self, 
                num_particles=5000, 
                max_timesteps=10,
                boundary=(40,40), 
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
        
        self.dim = 2
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

        self.grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))

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

        self.position_deltas = ti.Vector(self.dim, dt=ti.f32)
        # 0: x-pos, 1: y-pos of bot left
        self.board_states = ti.Vector(3, dt=ti.f32)
        # 0: weight, 1: height
        self.board_dims = ti.Vector(2, dt=ti.f32)

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

        grid_snode = ti.root.dense(ti.ij, self.grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.k, self.max_num_particles_per_cell).place(self.grid2particles)

        nb_node = ti.root.dense(ti.i, self.num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, self.max_num_neighbors).place(self.particle_neighbors)

        ti.root.place(self.board_states)
        ti.root.place(self.board_dims)
        
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

    # Python-scope to control emission and color setting
    def emit_particles(self, num, frame, init_pos, init_vel):
        if frame+1 >= self.max_timesteps:
            return
        if self.num_active[frame] >= self.num_particles:
            return

        # color_ind = self.num_active[frame] / self.num_particles
        # color_ind = round(0.2 * round(color_ind/0.2) , 1)
        # color = self.color_map(color_ind)

        # x = int(np.ceil(np.sqrt(num)))
        # xy = np.mgrid[-1.2:1.2:x*1j, -1.2:1.2:x*1j].reshape(2,-1).T

        for i in range(num):
            if self.num_active[frame] < self.num_particles:
                offset = np.array([0,0])
                self.emit_particles_kernel(frame, init_pos + offset, init_vel)
                #self.particle_rgba[self.num_active[frame],:] = np.array(color)
                self.particle_active[frame, self.num_active[frame]] = 1
                self.num_active[frame] += 1     


    @ti.kernel
    def init_board(self):
        self.board_states[None] = ti.Vector([10, 2, 0.0])
        self.board_dims[None] = ti.Vector([2, 10])
    
    @ti.func
    def poly6_value(self, s, h):
        result = 0.0
        if 0 < s and s < h:
            x = (h * h - s * s) / (h * h * h)
            result = self.poly6_factor * x * x * x
        return result

    @ti.func
    def spiky_gradient(self, r, h):
        result = ti.Vector([0.0, 0.0])
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
            ] and c[1] < self.grid_size[1]

    @ti.func
    def confine_position_to_boundary(self, p):
        # Global boundaries
        bmin = self.particle_radius_in_world
        bmax = ti.Vector([self.boundary[0], self.boundary[1]]) - self.particle_radius_in_world

        for i in ti.static(range(self.dim)):
            # Use randomness to prevent particles from sticking into each other after clamping
            if p[i] <= bmin:
                p[i] = bmin + self.epsilon * ti.random()
            elif bmax[i] <= p[i]:
                p[i] = bmax[i] - self.epsilon * ti.random()
        return p

    @ti.func
    def confine_position_to_board(self, p0, p1, suction):
        board_left = self.board_states[None][0]
        board_right = self.board_states[None][0] + self.board_dims[None][0]
        board_bot = self.board_states[None][1]
        board_top = self.board_states[None][1] + self.board_dims[None][1]

        p_proj = ti.Vector([p1[0], p1[1]])

        # If particle is in the interior of the board rect
        if p1[0] >= board_left and p1[0] <= board_right and p1[1] >= board_bot and p1[1] <= board_top:
            # Regular projection, the projection is based on particle's previous location
            if p0[0] <= board_left + 0.5:
                p_proj[0] = board_left - self.epsilon
            elif p0[0] >= board_right - 0.5:
                p_proj[0] = board_right + self.epsilon
            if p0[1] <= board_bot + 0.5:
                p_proj[1] = board_bot - self.epsilon
            elif p0[1] >= board_top - 0.5:
                p_proj[1] = board_top + self.epsilon
            
            if suction == 1:
                p_proj = ti.Vector([self.boundary[0]+100, self.boundary[1]+100])
            
        return p_proj


    @ti.kernel
    def move_board(self):
        # probably more accurate to exert force on particles according to hooke's law.
        b = self.board_states[None]
        b[2] += 1.0
        period = 200
        vel_strength = 2.0
        if b[2] >= 2 * period:
            b[2] = 0
        b[0] += ti.sin(b[2] * np.pi / period) * vel_strength * self.time_delta
        self.board_states[None] = b

    @ti.kernel
    def apply_suction(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i]:
                pos_i = self.positions[frame-1, i]

                board_left = self.board_states[None][0]
                board_right = self.board_states[None][0] + self.board_dims[None][0]
                board_bot = self.board_states[None][1]
                board_top = self.board_states[None][1] + self.board_dims[None][1]

                # If particle is within some area, take it out of simulation and move it to dummy location
                if pos_i[0] >= board_left and pos_i[0] <= board_right and pos_i[1] >= board_bot - 0.5:
                    self.particle_active[frame,i] = 0
                    self.positions[frame, i] = ti.Vector([self.boundary[0]+100, self.boundary[1]+100])

    @ti.kernel
    def apply_gravity_within_boundary(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                g = ti.Vector([0.0, -9.8])
                pos, vel = self.positions[frame-1,i], self.velocities[frame-1,i]
                vel += g * self.time_delta
                pos += vel * self.time_delta
                #self.positions[frame,i] = self.confine_position_to_boundary(pos)
                pos = self.confine_position_to_board(self.positions[frame-1,i], pos, 0)
                self.total_pos_delta[i] = self.confine_position_to_boundary(pos) - self.positions[frame-1,i]

    @ti.kernel
    def confine_to_boundary(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos = self.positions[frame-1,i] + self.total_pos_delta[i]
                #self.positions[frame,i] = self.confine_position_to_boundary(pos)
                pos = self.confine_position_to_board(self.positions[frame-1,i], pos, 0)
                self.total_pos_delta[i] = self.confine_position_to_boundary(pos) - self.positions[frame-1,i]

    @ti.kernel
    def update_grid(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                cell = self.get_cell(self.positions[frame-1,i] + self.total_pos_delta[i])
                # ti.Vector doesn't seem to support unpacking yet
                # but we can directly use int Vectors as indices

                # From pbf2d
                offs = self.grid_num_particles[cell].atomic_add(1)
                self.grid2particles[cell, offs] = i

                # Make grid2particles and indicator tensor
                #self.grid2particles[cell, i] = 1
        

    @ti.kernel
    def find_particle_neighbors(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos_i = self.positions[frame-1,i] + self.total_pos_delta[i]
                cell = self.get_cell(pos_i)
                nb_i = 0
                for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                    cell_to_check = cell + offs
                    if self.is_in_grid(cell_to_check):
                        for j in range(self.grid_num_particles[cell_to_check]):
                            p_j = self.grid2particles[cell_to_check, j]
                            if nb_i < self.max_num_neighbors and p_j != i:
                                dist = (pos_i - (self.positions[frame-1,p_j]+self.total_pos_delta[p_j])).norm()
                                if (dist < self.neighbor_radius):
                                    self.particle_neighbors[i, nb_i] = p_j
                                    nb_i += 1
                self.particle_num_neighbors[i] = nb_i


    @ti.kernel
    def compute_lambdas(self, frame: ti.i32):
        # Eq (8) ~ (11)
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos_i = self.positions[frame-1,i] + self.total_pos_delta[i]

                grad_i = ti.Vector([0.0, 0.0])
                sum_gradient_sqr = 0.0
                density_constraint = 0.0

                for j in range(self.particle_num_neighbors[i]):
                    p_j = self.particle_neighbors[i, j]
                    # TODO: does taichi supports break?
                    if p_j >= 0:
                        pos_ji = pos_i - (self.positions[frame-1,p_j] + self.total_pos_delta[p_j])
                        # grad_j is gradient for jth neighbor
                        # with respect to p_i
                        grad_j = self.spiky_gradient(pos_ji, self.h)
                        # All these accumulations do not work for gradient
                        grad_i += grad_j
                        sum_gradient_sqr += grad_j.dot(grad_j)
                        # # Eq(2)
                        density_constraint += self.poly6_value(pos_ji.norm(), self.h)

                # Eq(1)
                density_constraint = (self.mass * density_constraint / self.rho0) - 1.0

                sum_gradient_sqr += grad_i.dot(grad_i)
                self.lambdas[i] = (-density_constraint) / (sum_gradient_sqr +
                                                        self.lambda_epsilon)


    @ti.kernel
    def compute_position_deltas(self, frame: ti.i32):
        # Eq(12), (14)
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos_i = self.positions[frame-1,i] + self.total_pos_delta[i]
                lambda_i = self.lambdas[i]

                pos_delta_i = ti.Vector([0.0, 0.0])
                for j in range(self.particle_num_neighbors[i]):
                    p_j = self.particle_neighbors[i, j]
                    # TODO: does taichi supports break?
                    if p_j >= 0:
                        lambda_j = self.lambdas[p_j]
                        pos_ji = pos_i - (self.positions[frame-1,p_j] + self.total_pos_delta[p_j])
                        scorr_ij = self.compute_scorr(pos_ji)
                        pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                            self.spiky_gradient(pos_ji, self.h)

                pos_delta_i /= self.rho0
                self.position_deltas[i] = pos_delta_i

    @ti.kernel
    def apply_position_deltas(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                #self.positions[frame,i] += self.position_deltas[i]
                self.total_pos_delta[i] += self.position_deltas[i]

    @ti.kernel
    def update_velocities(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                self.velocities[frame,i] = (self.positions[frame,i] - self.positions[frame-1,i]) / self.time_delta

    @ti.kernel
    def apply_XSPH_viscosity(self, frame: ti.i32):
        # Eq (17)
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                pos_i = self.positions[frame,i]
                vel_i = self.velocities[frame,i]
                v_delta_i = ti.Vector([0.0, 0.0])

                for j in range(self.particle_num_neighbors[i]):
                    p_j = self.particle_neighbors[i, j]
                    pos_ji = pos_i - self.positions[frame,p_j]
                    vel_ij = self.velocities[frame,p_j] - vel_i
                    if p_j >= 0:
                        pos_ji = pos_i - self.positions[frame,p_j]
                        v_delta_i += vel_ij * self.poly6_value(pos_ji.norm(), self.h)

                self.velocities[frame,i] += self.viscosity_c * v_delta_i


    @ti.kernel
    def copy_particle_active(self, frame: ti.i32):
        for i in range(self.num_particles):
            self.particle_active[frame,i] = self.particle_active[frame-1,i]

    @ti.kernel
    def clear_total_pos_delta(self):
        for i in range(self.num_particles):
            for j in ti.static(range(self.dim)):
                self.total_pos_delta[i][j] = 0

    @ti.kernel
    def apply_total_pos_delta(self, frame: ti.i32):
        for i in range(self.num_particles):
            if self.particle_active[frame,i] == 1:
                self.positions[frame,i] += self.positions[frame-1,i] + self.total_pos_delta[i]


    #@ti.complex_kernel
    def run_pbf(self, frame: ti.i32):
        self.clear_total_pos_delta()

        self.num_active[frame] = self.num_active[frame-1]
        self.copy_particle_active(frame)

        self.apply_suction(frame)
        
        self.apply_gravity_within_boundary(frame)

        self.grid_num_particles.fill(0)
        self.particle_neighbors.fill(-1)
        self.grid2particles.fill(0)
        self.update_grid(frame)

        self.particle_num_neighbors.fill(0)
        self.find_particle_neighbors(frame)

        for _ in range(self.pbf_num_iters):

            self.compute_lambdas(frame)
            self.compute_position_deltas(frame)
            self.apply_position_deltas(frame)
        
        self.confine_to_boundary(frame)
        self.apply_total_pos_delta(frame)

        self.update_velocities(frame)
        self.apply_XSPH_viscosity(frame)


    def render(self,frame):
        #result_dir = "./viz_results/x_z_emit/frames"
        if self.gui is None and self.gui.running:
            print("Can't render without running GUI.")
            return
        canvas = self.gui.canvas
        canvas.clear(self.bg_color)
        pos_np = self.positions.to_numpy()[frame,:,:]
        active_np = self.particle_active.to_numpy()[frame,:]

        # Particles
        for i in range(pos_np.shape[0]):
            if active_np[i] == 1:
                for j in range(self.dim):
                    pos_np[i,j] *= self.render_scaling / self.render_res[j]
                self.gui.circle(pos_np[i,[0,1]], radius=self.particle_radius, color=self.particle_color)

        # Boundary rect
        canvas.rect(ti.vec(0, 0), 
                    ti.vec(1.0,1.0)
                    ).radius(1.5).color(self.boundary_color).close().finish()

        # Board rect
        botLeftX = self.board_states[None][0] / self.boundary[0]
        botLeftY = self.board_states[None][1] / self.boundary[1]
        topRightX = (self.board_states[None][0]+self.board_dims[None][0]) / self.boundary[0]
        topRightY = (self.board_states[None][1]+self.board_dims[None][1]) / self.boundary[1]
        canvas.rect(ti.vec(botLeftX, botLeftY), ti.vec(topRightX, topRightY)
                    ).radius(1.5).color(self.boundary_color).close().finish()
        
        if self.do_render_save:
            self.gui.show("{}/{:04d}.png".format(self.render_save_dir,frame))
        else:
            self.gui.show()

    def print_stats(self):
        print('PBF stats:')
        num = self.grid_num_particles.to_numpy()
        avg, max = np.mean(num), np.max(num)
        print(f'  #particles per cell: avg={avg:.2f} max={max}')
        num = self.particle_num_neighbors.to_numpy()
        avg, max = np.mean(num), np.max(num)
        print(f'  #neighbors per particle: avg={avg:.2f} max={max}')

    def save_ply(self,frame):
        ply_writer = ti.PLYWriter(num_vertices=self.num_active[None])
        pos_np = self.positions.to_numpy()[frame,:,:]
        active_np = self.particle_active.to_numpy()[frame,:]
        save_inds = active_np == 1
        ply_writer.add_vertex_pos(pos_np[save_inds, 0], pos_np[save_inds, 1], pos_np[save_inds, 2])
        ply_writer.add_vertex_rgba(self.particle_rgba[save_inds,0],
                                    self.particle_rgba[save_inds,1],
                                    self.particle_rgba[save_inds,2],
                                    self.particle_rgba[save_inds,3])
        #series_prefix = "./viz_results/3D/colors/frame.ply"
        ply_writer.export_frame_ascii(frame+1, self.ply_save_prefix)


    def step(self,frame):
        # No more sim after max_timesteps
        # Do frame+1 because run_pbf computes the particles for the next timestep
        # based on this time step's data
        if frame < self.max_timesteps:
            # Render first because emitted particles are initialization
            if self.do_render:
                self.render(frame)
            if self.do_ply_save:
                self.save_ply(frame)
            if self.do_print_stats:
                self.print_counter += 1
                if self.print_counter == self.print_frequency:
                    self.print_stats()
                    self.print_counter = 0

        if frame+1 < self.max_timesteps:
            self.move_board()
            self.run_pbf(frame+1)


        # self.loss[None] = 0
        # with ti.Tape(loss=self.loss):
        #     self.compute_loss()
        # print(self.old_positions.grad)
        # print("Alive")

    def set_target(self, target):
        for i in range(self.dim):
            self.target[None][i] = target[i]

    @ti.kernel
    def compute_loss(self, frame: ti.i32):
        #for i in range(self.max_timesteps):
        # for j in ti.static(range(self.dim)):
        #     self.loss[None] += self.positions[frame,0][j]
        # for i in range(self.num_particles):
        #     #if self.particle_active[frame, i] == 1:
        #     self.loss[None] = 1/2 * (15 - self.positions[frame, i][0])**2
        for j in ti.static(range(self.dim)):
            self.loss[None] += 1/2 * (self.positions[frame, 0][j] - self.target[None][j])**2


    