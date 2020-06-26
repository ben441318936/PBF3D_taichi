import taichi as ti
import numpy as np
from DiffFluidSim3D import DiffFluidSim3D

ti.init(arch=ti.gpu)
#gui = ti.GUI('PBF3D', (400,400))

#sim = DiffFluidSim3D(num_particles=1, max_timesteps=2, gui=None, do_render=False, do_print_stats=False, render_res=(400,400))
#sim = FluidSim3D(do_ply_save=True, ply_save_prefix="./viz_results/3D/colors/frame.ply")
# sim.initialize()

#i = 0

# for j in range(600):
#     if i < 312:
#         sim.emit_particles(16,np.array([10,10,10]),np.array([10,0,0]))
#         i += 1
#     sim.step(j)


# while gui.running:
#     sim.emit_particles(1, i, np.array([10,10,10]), np.array([10,0,0]))
#     sim.step(i)
#     i += 1

#     # if i >= sim.max_timesteps:
#     #     # for j in range(sim.max_timesteps):
#     #     #     print(sim.positions[j,0][0])
#     #     sim.loss[None] = 0
#     #     with ti.Tape(loss=sim.loss):
#     #         sim.compute_loss(sim.max_timesteps-1)
#     #     print(sim.positions.grad[0,0][0])
#     #     break

max_steps = 15

sim = DiffFluidSim3D(num_particles=3, max_timesteps=max_steps, do_print_stats=False)
sim.initialize()

target = np.array([30, 15, 15])

initial_vel = np.array([10.0,0.0,0.0])

for k in range(100):
    print("Gradient descent iter {}".format(k))
    sim.initialize()
    sim.set_target(target)
    steps = max_steps-1
    sim.loss[None] = 0
    sim.emit_particles(1, 0, np.array([10,10,10]), initial_vel)

    with ti.Tape(loss=sim.loss):
        for i in range(steps):        
            #sim.step(i)
            sim.run_pbf(i+1)
            #print("X pos {}: ".format(i), sim.positions[i,0][0])
            #print("X vel {}: ".format(i), sim.velocities[i,0][0])
        sim.compute_loss(10)
    #print("X pos final: ", sim.positions[10,0][0])
    print("Pos final: ", sim.positions[10,0][0], sim.positions[10,0][1], sim.positions[10,0][2])
    print("Loss: ", sim.loss[None])
    #print("Grad to initial pos: ", sim.positions.grad[0,0][0], sim.positions.grad[0,0][1], sim.positions.grad[0,0][2])
    print("Grad to initial vel: ", sim.velocities.grad[0,0][0], sim.velocities.grad[0,0][1], sim.velocities.grad[0,0][2])

    if sim.loss[None] <= 1e-5:
        break

    initial_vel = initial_vel - 5e-4 * np.array([sim.velocities.grad[0,0][0], sim.velocities.grad[0,0][1], sim.velocities.grad[0,0][2]])
    print("New initial_vel: ", initial_vel)