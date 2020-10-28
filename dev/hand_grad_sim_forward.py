from hand_grad_sim_3D_test import HandGradSim3D
import numpy as np
import pickle

import argparse


parser = argparse.ArgumentParser(description="Pick the emission point")
parser.add_argument("exp_num", type=int)

args = parser.parse_args()

exp_num = 520 + args.exp_num

emit_pos = np.load("./cavity1_emit_pos.npy")[args.exp_num]
emit_vel = np.load("./cavity1_emit_vel.npy")[args.exp_num]

time = 400

sim = HandGradSim3D(max_timesteps=time, num_particles=500, do_save_npy=True, do_emit=True)
sim.exp = "exp{}".format(exp_num)
sim.make_save_paths()
sim.set_emit(emit_pos, emit_vel)

initial_pos0 = np.array([10.0, 10.0, 10.0])
initial_vel0 = np.array([10.0, 0.0, 0.0])

trajectory = np.load("./cavity2_manual_path_right.npy")

# board_states = np.zeros((sim.max_timesteps,sim.dim))
# for i in range(sim.max_timesteps):
#     board_states[i,:] = np.array([10.0, 20.0])

# tool_centers = np.zeros((sim.max_timesteps,sim.dim))
# for i in range(sim.max_timesteps):
#     tool_centers[i,:] = np.array([16.0, 7.0])

# tool_thetas = np.zeros((sim.max_timesteps,))
# for i in range(sim.max_timesteps):
#     tool_thetas[i] = i * 2 * np.pi / sim.max_timesteps

# best_states_path = "./states/set2/best_states_1.obj"
# with open(best_states_path, "rb") as f:
#     board_states = pickle.load(f)

# iter_states_path = "./states/iter_states.obj"
# with open(iter_states_path, "rb") as f:
#     iter_states = pickle.load(f)

# board_states = iter_states["iter1000"]

board_states = np.zeros((time, 3))
for i in range(100):
    board_states[i,:] = np.array([1, 20, 15])
for i in range(100, time):
    # board_states[i,:] = np.array([2, 0.5, 2])
    board_states[i,:] = trajectory[i-200]
    # board_states[i,:] = np.array([1, 20, 15])

sim.initialize(tool_states=board_states)

# init_pos_path = "./states/init_pos.obj"
# with open(init_pos_path, "rb") as f:
#     init_pos = pickle.load(f)

# init_vel_path = "./states/init_vel.obj"
# with open(init_vel_path, "rb") as f:
#     init_vel = pickle.load(f)  

# sim.emit_particles(100, 0, init_pos, init_vel)

sim.forward()

# pos = sim.positions.to_numpy()
# pos = pos[-1,:,:]
# print(pos.shape)

# pos_path = "./states/init_pos.obj"
# with open(pos_path, "w+b") as f:
#     pickle.dump(pos, f)

# vel = sim.velocities.to_numpy()
# vel = vel[-1,:,:]
# print(vel.shape)

# vel_path = "./states/init_vel.obj"
# with open(vel_path, "w+b") as f:
#     pickle.dump(vel, f) 