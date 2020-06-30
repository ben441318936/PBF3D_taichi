import taichi as ti
from DiffFluidSim3D import DiffFluidSim3D
import numpy as np



gui = ti.GUI('PBF3D', (400,400))

sim = DiffFluidSim3D(num_particles=100, max_timesteps=500, gui=gui, do_render=True, do_print_stats=False, render_res=(400,400))
sim.initialize()

for i in range(500):
    sim.emit_particles(1, i, np.array([10,10,10]), np.array([10,0,0]))
    sim.step(i)


