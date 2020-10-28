import argparse

from MPC_new import MPC
from MainSim import MainSim
from AuxiliarySim import AuxiliarySim
from hand_grad_sim_3D_test import HandGradSim3D
import numpy as np

parser = argparse.ArgumentParser(description="Pick the emission point")
parser.add_argument("exp_num", type=int)

args = parser.parse_args()

exp_num = 520 + args.exp_num

emit_pos = np.load("./cavity1_emit_pos.npy")[args.exp_num]
emit_vel = np.load("./cavity1_emit_vel.npy")[args.exp_num]

real_world_max_time = 500
real_world_warm_up_time = 200
max_particles = real_world_max_time
main_sim_max_time = 100
aux_sim_max_time = 10
real_world_exp_num = exp_num

real_world = HandGradSim3D(max_timesteps=real_world_max_time, num_particles=max_particles, do_save_npy=True, do_emit=True)
real_world.set_emit(emit_pos, emit_vel)
real_world.exp = "exp{}".format(real_world_exp_num)
real_world.make_save_paths()

mpc = MPC(main_sim_horizon=main_sim_max_time, aux_sim_horizon=aux_sim_max_time, num_particles=max_particles, do_emit=True)
mpc.set_emit(emit_pos, emit_vel)

real_world.initialize()
real_world.init_step()

# Run the main sim for some time to fill up particles
for i in range(1,real_world_warm_up_time):
    real_world.take_action(i, np.array([15.0, 20.0, 15.0]))

# Find optimal initial suction point
# Extract info from real life
real_life_state = mpc.extract_sim_states(real_world, real_world_warm_up_time)

# # Try an initial suction point
# mpc.init_tool_state = np.array([1, 0.5, 13])
# mpc.set_exp("exp45")
# mpc.init_main_sim(real_life_state)
# mpc.run_MPC()

# # Try an other initial suction point
# mpc.init_tool_state = np.array([13, 0.5, 13])
# mpc.set_exp("exp46")
# mpc.init_main_sim(real_life_state)
# mpc.run_MPC()

num_samples = 10
sample_positions = []
remaining_particles = []

particle_pos = real_life_state[1]
particle_status = real_life_state[3]

eligible_inds = np.nonzero(particle_status == 1)

selected = np.random.choice(eligible_inds[0], num_samples, replace=False)

for i in range(num_samples):
    print(exp_num)
    print("Sample {}".format(i))
    sample_positions.append(particle_pos[selected[i], :])
    mpc.init_tool_state = sample_positions[i]
    # mpc.set_exp("exp{}".format(real_world_exp_num+1+i))
    mpc.init_main_sim(real_life_state)
    mpc.run_MPC()
    # Check number of remaining particles
    particle_status = mpc.main_sim.extract_sim_states(main_sim_max_time)[3]
    remain = np.sum(particle_status == 1)
    remaining_particles.append(remain)

print("Selected indices:", selected)
print("Positions:", sample_positions)
print("Remaining:", remaining_particles)

# sample_positions = [np.array([0.3000026 , 0.30000854, 0.7581406 ]), 
#                     np.array([12.589552  ,  0.30001146,  9.430138  ]), 
#                     np.array([11.388903 ,  0.3000017, 13.429737 ]), 
#                     np.array([0.3000044, 0.8959166, 0.30001  ]), 
#                     np.array([10.966219  ,  0.30000904, 12.940251  ]), 
#                     np.array([ 0.31517318,  0.9127429 , 10.206581  ]), 
#                     np.array([4.041043 , 0.3013001, 2.0956604]), 
#                     np.array([2.0462039, 0.8869226, 3.496094 ]), 
#                     np.array([6.233744  , 0.30059028, 1.671679  ]), 
#                     np.array([ 2.7750716 ,  0.30081674, 10.928317  ])]

# remaining_particles = [100, 162, 167, 114, 176, 80, 107, 130, 175, 67]

best_start = sample_positions[np.argmin(np.array(remaining_particles))]

tool = best_start

print("Finish MPC in real world")

# Then we use this start point to do the rest of the MPC
for i in range(real_world_warm_up_time, real_world_max_time):
    real_life_state = mpc.extract_sim_states(real_world, i)
    tool = mpc.MPC_aux_step(tool, real_life_state)
    real_world.take_action(i, tool)

