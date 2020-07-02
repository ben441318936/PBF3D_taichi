from hand_grad_sim import HandGradSim
import numpy as np

sim = HandGradSim()

loss = 10
k = 0

initial_pos = np.array([10.0, 10.0])
initial_vel = np.array([9.0, 0.0])

while loss > 1e-5 and k <= 100:

    print("Gradient descent iter:", k)
    k += 1

    sim.initialize()
    sim.place_particle(0, 0, initial_pos, initial_vel)
    sim.forward()
    print("Final pos:", sim.positions[sim.max_timesteps-1,0][0], sim.positions[sim.max_timesteps-1,0][1])
    print("Loss:", sim.loss[None])
    loss = sim.loss[None]
    sim.backward()
    print("Grad to initial pos:", sim.positions.grad[0,0][0], sim.positions.grad[0,0][1])
    print("Grad to initial vel:", sim.velocities.grad[0,0][0], sim.velocities.grad[0,0][1])

    initial_vel -= 1e-1 * np.array([sim.velocities.grad[0,0][0], sim.velocities.grad[0,0][1]])
    print("New initial vel:", initial_vel)

    # initial_pos -= 5e-1 * np.array([sim.positions.grad[0,0][0], sim.positions.grad[0,0][1]])
    # print("New initial pos:", initial_pos)