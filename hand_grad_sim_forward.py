from hand_grad_sim import HandGradSim
import numpy as np
import pickle

sim = HandGradSim()

initial_pos0 = np.array([10.0, 10.0])
initial_vel0 = np.array([10.0, 0.0])

# board_states = np.zeros((sim.max_timesteps,sim.dim))
# for i in range(sim.max_timesteps):
#     board_states[i,:] = np.array([10.0, 20.0])

# best_states_path = "./states/best_states_new.obj"
# with open(best_states_path, "rb") as f:
#     board_states = pickle.load(f)

iter_states_path = "./states/iter_states.obj"
with open(iter_states_path, "rb") as f:
    iter_states = pickle.load(f)

board_states = iter_states["iter1000"]

sim.initialize(board_states)

# for i in range(sim.num_particles):
#     sim.place_particle(0, i, np.array([i/4, 1]), initial_vel0)

sim.forward()