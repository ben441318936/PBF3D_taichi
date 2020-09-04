from MPC import MPC
import numpy as np

test = MPC(main_sim_horizon=200, aux_sim_horizon=10, num_particles=600, init_point=np.array([1, 0.5, 13]), warm_up_steps=99, do_save_npy=True, do_emit=True)

test.start_actual_sim()
test.warm_up_actual_sim()
test.run_MPC()


