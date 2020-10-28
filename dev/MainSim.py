from hand_grad_sim_3D_test import HandGradSim3D
import numpy as np

class MainSim:
    def __init__(self, horizon=100, num_particles=600, init_sim_states=None, do_emit=True):
        self.horizon = horizon

        self.main_sim = HandGradSim3D(max_timesteps=horizon, num_particles=num_particles, do_save_npy=False, do_emit=do_emit)

        self.best_point = np.array([0.0, 0.0, 0.0])

        self.init_sim_states = init_sim_states

    def set_emit(self, emit_pos, emit_vel):
        self.main_sim.set_emit(emit_pos, emit_vel)

    def init_sim(self):
        self.main_sim.initialize()
        self.main_sim.emit_particles(self.init_sim_states[0], 0, self.init_sim_states[1], self.init_sim_states[2], self.init_sim_states[3])
        self.main_sim.init_step()

    def extract_sim_states(self, t):
        part_pos = self.main_sim.positions.to_numpy()[t-1,:,:]
        part_vel = self.main_sim.velocities.to_numpy()[t-1,:,:]
        part_active = self.main_sim.particle_active.to_numpy()[t-1,:]

        actives = np.where(np.logical_or(part_active==1, part_active==2))[0]
        active_status = part_active[actives]
        active_pos = part_pos[actives, :]
        active_vel = part_vel[actives, :]

        return (len(actives), active_pos, active_vel, active_status)

    def take_action(self, t, new_point):
        self.main_sim.take_action(t, new_point)
        

    

        


    