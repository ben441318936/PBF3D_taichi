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

steps = max_steps-1
sim.loss[None] = 0
sim.emit_particles(1, 0, np.array([10,10,1.7]), np.array([10,0,0]))

with ti.Tape(loss=sim.loss):
    for i in range(steps):        
        #sim.step(i)
        sim.run_pbf(i+1)
        print("Z pos {}: ".format(i), sim.positions[i,0][2])
    sim.compute_loss(10)
print("Z pos final: ", sim.positions[steps,0][2])
print("Loss: ", sim.loss[None])
print("Grad to initial pos: ", sim.positions.grad[0,0][0], sim.positions.grad[0,0][1], sim.positions.grad[0,0][2])
print("Grad to initial pos: ", sim.positions.grad[0,1][0], sim.positions.grad[0,1][1], sim.positions.grad[0,1][2])
