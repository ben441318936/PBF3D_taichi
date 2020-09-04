from hand_grad_sim_3D import HandGradSim3D
import numpy as np

class MPC:
    def __init__(self, main_sim_horizon=200, aux_sim_horizon=10, num_particles=600, init_point=np.array([1, 0.5, 13]), warm_up_steps=99, do_save_npy=True, do_emit=True):
        self.actual_sim = HandGradSim3D(max_timesteps=main_sim_horizon, num_particles=num_particles, do_save_npy=do_save_npy, do_emit=do_emit)
        self.aux_sim = HandGradSim3D(max_timesteps=aux_sim_horizon, num_particles=num_particles, do_save_npy=False, do_emit=do_emit)

        self.final_tool_trajectory = 100*np.ones((self.actual_sim.max_timesteps, self.actual_sim.dim))

        self.init_tool_states = np.zeros((self.aux_sim.max_timesteps, self.aux_sim.dim))

        for i in range(self.aux_sim.max_timesteps):
            self.init_tool_states[i,:] = init_point
        
        self.best_states = self.init_tool_states.copy()
        self.best_point = self.best_states[1,:]

        self.warm_up_steps = warm_up_steps

    def start_actual_sim(self):
        # Start actual sim
        self.actual_sim.initialize()
        self.actual_sim.init_step()

    def warm_up_actual_sim(self):
        # Run the main sim for some time to fill up particles
        for i in range(1,1+self.warm_up_steps):
            self.actual_sim.take_action(i, np.array([10.0, 20.0, 10.0]))

    def run_MPC(self):
        for i in range(self.warm_up_steps+1, self.actual_sim.max_timesteps):
            self.MPC_loop(i)

    def MPC_loop(self, i, max_iters=21):
        print("Finding action", i)
        # log_file.write("Finding action {}\n".format(i))
        # actual_sim.take_action(i,np.array([10.0, 20.0]))

        # Read out particle states at the most recent frame
        part_pos = self.actual_sim.positions.to_numpy()[i-1,:,:]
        part_vel = self.actual_sim.velocities.to_numpy()[i-1,:,:]
        part_active = self.actual_sim.particle_active.to_numpy()[i-1,:]
        part_num_active = self.actual_sim.num_active.to_numpy()[i-1]
        part_num_suctioned = self.actual_sim.num_suctioned.to_numpy()[i-1]

        # if part_num_active > 0:

        # Do gradient descent using aux sim
        best_loss = 1e7
        best_iter = 0 
        loss = best_loss
        k = 0
        lr = 1e-1

        old_best_point = self.best_point.copy()

        while loss > 1e-2 and k < max_iters:
            # Clear the aux sim
            # print("Iter", k)
            # log_file.write("Iter {}\n".format(k))
            self.aux_sim.initialize(self.init_tool_states)

            # Place active particles into aux sim
            actives = np.where(np.logical_or(part_active==1, part_active==2))[0]
            active_status = part_active[actives]
            active_pos = part_pos[actives, :]
            active_vel = part_vel[actives, :]
            self.aux_sim.emit_particles(len(actives), 0, active_pos, active_vel, active_status)

            # print(len(actives))

            self.aux_sim.forward()
            loss = self.aux_sim.loss[None]
            # print(loss)
        
            if loss <= best_loss:
                best_loss = loss
                best_iter = k
                self.best_states = self.init_tool_states.copy()

            self.aux_sim.backward()
            tool_state_grads = self.aux_sim.tool_states.grad.to_numpy()
            # print(tool_state_grads)
            # log_file.write(np.array_str(tool_state_grads)+"\n")

            for l in range(tool_state_grads.shape[0]):
                m = np.max(np.abs(tool_state_grads[l,:]))
                if m >= 1:
                    tool_state_grads[l,:] = tool_state_grads[l,:] / m

            # tool_state_grads = np.clip(tool_state_grads, -10, 10)
            # print(tool_state_grads)

            self.init_tool_states -= lr * tool_state_grads
            for l in range(self.init_tool_states.shape[0]):
                self.init_tool_states[l,:] = self.aux_sim.confine_tool_to_boundary(self.init_tool_states[l,:])
            # print(init_tool_states)

            k += 1

            if k % 20 == 0:
                lr *= 0.95

        # Project the solved point
        # print(best_states)
        self.best_point = self.best_states[1,:]
        dif = self.best_point - old_best_point
        m = np.max(np.abs(dif))
        c = 0.5
        if m >= c:
            dif = dif / m * c
        self.best_point = old_best_point + dif
        # print(best_point)
        self.best_point = self.actual_sim.confine_tool_to_boundary(self.best_point)
        # print(best_point)

        # Take the first step in the optimal trajectory will be used as init for future GD
        for j in range(0,self.aux_sim.max_timesteps):
            self.init_tool_states[j,:] = self.best_point

        # The first step in the optimal trajectory will be taken to the actual sim
        print(self.best_point)
        # print()
        # log_file.write("Best point: " + np.array_str(best_point) + "\n")
        # log_file.write("\n")
        self.actual_sim.take_action(i, self.best_point)
        self.final_tool_trajectory[i,:] = self.best_point




    
