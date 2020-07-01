import taichi as ti
from DiffFluidSim2D import DiffFluidSim2D
import numpy as np



gui = ti.GUI('PBF2D', (400,400))

sim = DiffFluidSim2D(num_particles=1000, max_timesteps=1500, gui=gui, do_render=True, do_render_save=True, render_save_dir="./viz_results/suction/frames",  do_print_stats=False, render_res=(400,400))
sim.initialize()

for i in range(1500):
    sim.emit_particles(1, i, np.array([2,2]), np.array([10,0]))
    sim.step(i)


