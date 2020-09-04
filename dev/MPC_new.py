from MainSim import MainSim
from AuxiliarySim import AuxiliarySim
import numpy as np

class MPC:
    def __init__(self, main_sim_horizon=200, aux_sim_horizon=10, num_particles=600, do_emit=True):
        # This is the long-term look-ahead for optimizing initial suction point in the outer loop
        self.main_sim = MainSim(horizon=main_sim_horizon, num_particles=num_particles, do_emit=True)
        # This is the short-term look-ahead for optimizing the next action in the inner loop
        self.aux_sim = AuxiliarySim(horizon=aux_sim_horizon, num_particles=num_particles, do_emit=True)
        # This tool state is the initial suction point
        self.init_tool_state = None

    # The sim states here should come from registration with real-world
    def init_main_sim(self, init_sim_states):
        self.main_sim.init_sim_states = init_sim_states
        self.main_sim.init_sim()

    # The tool states are optimized through MPC, we initialize the next one with the current solved best point
    # The sim states here should come from the main sim
    def init_aux_sim(self, init_tool_state, init_sim_states):
        self.aux_sim.set_init_tool_states(init_tool_state)
        self.aux_sim.init_sim_states = init_sim_states
        self.aux_sim.init_sim()

    def run_MPC(self):
        for i in range(1, self.main_sim.horizon):
            print("Finding action", i)
            sim_states = self.main_sim.extract_sim_states(i)

            if i == 1:
                # In the first iteration, we initialize the tool at the externally chosen suction point
                # print(self.init_tool_state)
                self.init_aux_sim(self.init_tool_state, sim_states)
            else:
                # Otherwise we use the previous solved best point as the init for the next step
                self.init_aux_sim(self.aux_sim.best_point, sim_states)

            # Then we do GD with the aux sim
            self.aux_sim.gradient_descent()
            best_point = self.aux_sim.best_point
            print(best_point)

            # Then take action 
            self.main_sim.take_action(i, best_point)

    # Helper function to extract info from the real life sim
    def extract_sim_states(self, sim, t):
        part_pos = sim.positions.to_numpy()[t-1,:,:]
        part_vel = sim.velocities.to_numpy()[t-1,:,:]
        part_active = sim.particle_active.to_numpy()[t-1,:]

        actives = np.where(np.logical_or(part_active==1, part_active==2))[0]
        active_status = part_active[actives]
        active_pos = part_pos[actives, :]
        active_vel = part_vel[actives, :]

        return (len(actives), active_pos, active_vel, active_status)




    
        
        