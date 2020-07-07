from hand_grad_sim import HandGradSim
import numpy as np

sim = HandGradSim()

initial_pos0 = np.array([10.0, 10.0])
initial_vel0 = np.array([10.0, 0.0])

sim.initialize()

# for i in range(sim.num_particles):
#     sim.place_particle(0, i, np.array([i/4, 1]), initial_vel0)

sim.forward()