from hand_grad_sim_3D_test import HandGradSim3D
import numpy as np

class AuxiliarySim:
    def __init__(self, horizon=10, num_particles=600, init_tool_state=None, init_sim_states=None, do_emit=True):
        self.aux_sim = HandGradSim3D(max_timesteps=horizon, num_particles=num_particles, do_save_npy=False, do_emit=do_emit)

        self.set_init_tool_states(init_tool_state)

        self.best_states = self.init_tool_states.copy()
        self.best_point = self.best_states[1,:]

        self.init_sim_states = init_sim_states

    def set_emit(self, emit_pos, emit_vel):
        self.aux_sim.set_emit(emit_pos, emit_vel)

    def set_init_tool_states(self, init_tool_state):
        self.init_tool_states = np.zeros((self.aux_sim.max_timesteps, self.aux_sim.dim))

        if init_tool_state is not None:
            # print("hi")
            for i in range(self.aux_sim.max_timesteps):
                self.init_tool_states[i,:] = init_tool_state
        else:
            for i in range(self.aux_sim.max_timesteps):
                self.init_tool_states[i,:] = self.aux_sim.boundary

        self.best_states = self.init_tool_states.copy()
        self.best_point = self.best_states[1,:]

    def init_sim(self):
        self.aux_sim.initialize(self.init_tool_states)
        self.aux_sim.emit_particles(self.init_sim_states[0], 0, self.init_sim_states[1], self.init_sim_states[2], self.init_sim_states[3])

    def gradient_descent(self, max_iters=21):
        # Do gradient descent using aux sim
        best_loss = 1e7
        loss = best_loss
        k = 0
        lr = 1e-1

        old_best_point = self.best_point.copy()

        while loss > 1e-2 and k < max_iters:
            # Init tool states change every iteration based on GD
            # Init sim states are constant until changed externally
            self.init_sim()

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

        self.best_point = self.aux_sim.confine_tool_to_boundary(self.best_states[1,:])

        dif = self.best_point - old_best_point
        m = np.max(np.abs(dif))
        c = 0.05
        if m >= c:
            dif = dif / m * c
        self.best_point = old_best_point + dif
        

    

        