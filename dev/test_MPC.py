from hand_grad_sim_3D import HandGradSim3D
import numpy as np
import pickle

actual_sim = HandGradSim3D(max_timesteps=200, num_particles=600, do_save_npy=True, do_emit=True)
aux_sim = HandGradSim3D(max_timesteps=10, num_particles=400, do_save_npy=False, do_emit=True)

final_tool_trajectory = 100*np.ones((actual_sim.max_timesteps, actual_sim.dim))

init_tool_states = np.zeros((aux_sim.max_timesteps, aux_sim.dim))
for i in range(aux_sim.max_timesteps):
    init_tool_states[i,:] = np.array([12.0, 0.5, 19.0])
best_states = init_tool_states.copy()
best_point = best_states[1,:]

# Start actual sim
actual_sim.initialize()
actual_sim.init_step()

# Run the main sim for some time to fill up particles
for i in range(1,100):
    actual_sim.take_action(i, np.array([10.0, 20.0, 10.0]))

for i in range(100,actual_sim.max_timesteps):
    print("Finding action", i)
    # actual_sim.take_action(i,np.array([10.0, 20.0]))

    # Read out particle states at the most recent frame
    part_pos = actual_sim.positions.to_numpy()[i-1,:,:]
    part_vel = actual_sim.velocities.to_numpy()[i-1,:,:]
    part_active = actual_sim.particle_active.to_numpy()[i-1,:]
    part_num_active = actual_sim.num_active.to_numpy()[i-1]
    part_num_suctioned = actual_sim.num_suctioned.to_numpy()[i-1]
    part_ages = actual_sim.particle_age.to_numpy()[i-1]

    if part_num_active - part_num_suctioned > 0:

        # Do gradient descent using aux sim
        best_loss = 1e7
        best_iter = 0   
        loss = best_loss
        k = 0
        lr = 1e-1

        while loss > 1e-2 and k < 21:
            # Clear the aux sim
            aux_sim.initialize(init_tool_states)

            # Place active particles into aux sim
            actives = np.where(part_active==1)[0]
            active_pos = part_pos[actives, :]
            active_vel = part_pos[actives, :]
            active_ages = part_ages[actives]
            aux_sim.emit_particles(len(actives), 0, active_pos, active_vel, active_ages)

            aux_sim.forward()
            loss = aux_sim.loss[None]
            # print(loss)
        
            if loss <= best_loss:
                best_loss = loss
                best_iter = k
                best_states = init_tool_states.copy()

            aux_sim.backward()
            tool_state_grads = aux_sim.board_states.grad.to_numpy()
            # print(tool_state_grads)

            for l in range(tool_state_grads.shape[0]):
                m = np.abs(np.max(tool_state_grads[l,:]))
                if m >= 1:
                    tool_state_grads[l,:] = tool_state_grads[l,:] / m

            # tool_state_grads = np.clip(tool_state_grads, -10, 10)
            # print(tool_state_grads)

            init_tool_states -= lr * tool_state_grads
            # print(init_tool_states)

            k += 1

            if k % 20 == 0:
                lr *= 0.95

        # Project the solved point
        # print(best_states)
        best_point = best_states[1,:]
        # print(best_point)
        best_point = actual_sim.confine_board_to_boundary(best_point)
        # print(best_point)

        # Take the first step in the optimal trajectory will be used as init for future GD
        for j in range(0,aux_sim.max_timesteps):
            init_tool_states[j,:] = best_point

    # The first step in the optimal trajectory will be taken to the actual sim
    print(best_point)
    actual_sim.take_action(i, best_point)
    final_tool_trajectory[i,:] = best_point

