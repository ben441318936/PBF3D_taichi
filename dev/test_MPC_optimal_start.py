from MPC_new import MPC
from MainSim import MainSim
from AuxiliarySim import AuxiliarySim
from hand_grad_sim_3D import HandGradSim3D
import numpy as np

real_world = HandGradSim3D(max_timesteps=200, num_particles=600, do_save_npy=False, do_emit=True)

mpc = MPC(main_sim_horizon=200, aux_sim_horizon=10, num_particles=600, do_emit=True)

real_world.initialize()
real_world.init_step()


# Run the main sim for some time to fill up particles
for i in range(1,100):
    real_world.take_action(i, np.array([10.0, 20.0, 10.0]))

# Find optimal initial suction point
# Initialize the main sim with real life
real_life_state = mpc.extract_sim_states(real_world, 100)
mpc.init_main_sim(real_life_state)
# Select an initial suction point
mpc.init_tool_state = np.array([1, 0.5, 13])

mpc.run_MPC()






