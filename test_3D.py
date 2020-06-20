import taichi as ti
import numpy as np
from FluidSim3D import FluidSim3D

ti.init(arch=ti.gpu)
gui = ti.GUI('PBF3D', (400,400))

sim = FluidSim3D(gui=gui, do_render=True, do_print_stats=True, render_res=(400,400))
sim.initialize()

i = 0

for j in range(600):
    if i < 500:
        if i%2==0:
            sim.emit_particles(10,np.array([10,10,10]),np.array([10,0,0]))
        i += 1
    sim.step(j)