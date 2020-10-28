from MPC_new import MPC
from MainSim import MainSim
from AuxiliarySim import AuxiliarySim
from hand_grad_sim_3D_test import HandGradSim3D
import numpy as np

real_world_max_time = 300
real_world_warm_up_time = 100
max_particles = real_world_max_time
main_sim_max_time = 100
aux_sim_max_time = 10
real_world_exp_num = 700

# Case 1
emit_pos = np.array([[0.7, 1.0, 0.5],[0.7, 1.0, 1.0],[0.7, 1.0, 1.5]])
emit_vel = np.array([[5.0, 0.0, 0.0],[5.0, 0.0, 0.0],[5.0, 0.0, 0.0]])

# Case 2
# emit_pos = np.array([[9.3, 1.0, 3.5],[9.3, 1.0, 4.0],[9.3, 1.0, 4.5]])
# emit_vel = np.array([[-5.0, 0.0, 0.0],[-5.0, 0.0, 0.0],[-5.0, 0.0, 0.0]])
                

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

tool = np.array([6.368543, 1.0612917, 7.186523]) # case 1

# tool = np.array([6.2543025 1.6525105 5.7532096]) # case 2

print("Finish MPC in real world")

# Then we use this start point to do the rest of the MPC
for i in range(real_world_warm_up_time, real_world_max_time):
    print("Step {}".format(i))
    real_life_state = mpc.extract_sim_states(real_world, i)
    tool = mpc.MPC_aux_step(tool, real_life_state)
    print(tool)
    real_world.take_action(i, tool)










