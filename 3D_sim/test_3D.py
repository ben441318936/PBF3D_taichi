import taichi as ti
import numpy as np
from FluidSim3D import FluidSim3D

ti.init(arch=ti.gpu)
gui = ti.GUI('PBF3D', (400,400))

sim = FluidSim3D(num_particles=100, gui=gui, do_render=True, do_print_stats=False, render_res=(400,400))
#sim = FluidSim3D(do_ply_save=True, ply_save_prefix="./viz_results/3D/colors/frame.ply")
sim.initialize()

i = 0

# for j in range(600):
#     if i < 312:
#         sim.emit_particles(16,np.array([10,10,10]),np.array([10,0,0]))
#         i += 1
#     sim.step(j)

while gui.running:
    sim.emit_particles(1,np.array([10,10,10]),np.array([10,0,0]))
    sim.step(0)