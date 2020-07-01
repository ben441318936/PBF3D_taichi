import taichi as ti
import numpy as np
import math

@ti.data_oriented
class HandGradSim:
    def __init__(self):
        self.dim = 2
        self.delta_t = 1.0 / 20.0
        self.max_timesteps = 20
        self.num_particles = 1

        self.target = ti.Vector(self.dim, ti.f32)

        self.positions = ti.Vector(self.dim, ti.f32)
        self.velocities = ti.Vector(self.dim, ti.f32)

        self.loss = ti.var(ti.f32)

        self.place_vars()

    def place_vars(self):
        ti.root.dense(ti.i, self.max_timesteps).dense(ti.j, self.num_particles).place(self.positions, self.velocities)
        ti.root.place(self.loss)
        ti.root.place(self.target)
        ti.root.lazy_grad()

    def initialize(self):
        self.positions.fill(0.0)
        self.velocities.fill(0.0)
        self.loss.fill(0.0)
        self.target[None][0] = 30
        self.target[None][1] = 15

    def clear_grads(self):
        self.positions.grad.fill(0.0)
        self.velocities.grad.fill(0.0)
        self.loss.grad.fill(0.0)

    @ti.kernel
    def place_particle(self, frame: ti.i32, i: ti.i32, p: ti.ext_arr(), v: ti.ext_arr()):
        for k in ti.static(range(self.dim)):
            self.positions[frame, i][k] = p[k]
            self.velocities[frame, i][k] = v[k]

    @ti.kernel
    def gravity_forward(self, frame: ti.i32):
        for i in range(self.num_particles):
            g = ti.Vector([0.0, -9.81])
            pos, vel = self.positions[frame-1,i], self.velocities[frame-1,i]
            vel += g * self.delta_t
            pos += vel * self.delta_t
            self.positions[frame,i] = pos

    @ti.kernel
    def gravity_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            self.positions.grad[frame-1,i] += self.positions.grad[frame,i] 
            self.velocities.grad[frame-1,i] += self.positions.grad[frame,i] * self.delta_t

    @ti.kernel
    def update_velocity_froward(self, frame: ti.i32):
        for i in range(self.num_particles):
            self.velocities[frame,i] = (self.positions[frame,i] - self.positions[frame-1,i]) / self.delta_t

    @ti.kernel
    def update_velocity_backward(self, frame: ti.i32):
        for i in range(self.num_particles):
            if frame == self.max_timesteps-1:
                pass
            else:
                self.positions.grad[frame,i] += self.velocities.grad[frame,i] / self.delta_t
                self.positions.grad[frame-1,i] += - self.velocities.grad[frame,i] / self.delta_t


    @ti.kernel
    def compute_loss_forward(self):
        for k in ti.static(range(self.dim)):
            self.loss[None] += 1/2 * (self.positions[self.max_timesteps-1,0][k] - self.target[None][k])**2

    @ti.kernel
    def compute_loss_backward(self):
        for k in ti.static(range(self.dim)):
            self.positions.grad[self.max_timesteps-1,0][k] += self.positions[self.max_timesteps-1,0][k] - self.target[None][k]

    def step_forward(self, frame):
        self.gravity_forward(frame)
        self.update_velocity_froward(frame)

    def backward_step(self, frame):
        self.update_velocity_backward(frame)
        self.gravity_backward(frame)

    
    def forward(self):
        for i in range(1,self.max_timesteps):
            self.step_forward(i)
        self.compute_loss_forward()

    def backward(self):
        self.clear_grads()
        self.compute_loss_backward()
        for i in reversed(range(1, self.max_timesteps)):
            self.backward_step(i)


