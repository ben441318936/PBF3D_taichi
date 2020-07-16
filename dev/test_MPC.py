from hand_grad_sim import HandGradSim
import numpy as np
import pickle

actual_sim = HandGradSim(max_timesteps=300, num_particles=200, do_render=True, do_emit=True)
aux_sim = HandGradSim(max_timesteps=10, num_particles=200, do_render=False, do_emit=True)

final_tool_trajectory = 100*np.ones((actual_sim.max_timesteps, actual_sim.dim))

# init_tool_states = np.zeros((aux_sim.max_timesteps, aux_sim.dim))
# for i in range(aux_sim.max_timesteps):
#     init_tool_states[i,:] = np.array([10.0, 20.0])
tool_centers = np.zeros((aux_sim.max_timesteps,aux_sim.dim))
for i in range(aux_sim.max_timesteps):
    tool_centers[i,:] = np.array([10.0, 20.0])
tool_thetas = np.zeros((aux_sim.max_timesteps,))
for i in range(aux_sim.max_timesteps):
    tool_thetas[i] = 0
best_states = tool_centers.copy()
best_point = best_states[1,:]

# Start actual sim
actual_sim.initialize()
actual_sim.init_step()

# Run the main sim for some time to fill up particles
for i in range(1,100):
    actual_sim.take_action(i, tool_centers[0,:])

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
        best_loss = 1e5
        best_iter = 0
        loss = best_loss
        k = 0
        lr = 5

        while loss > 1e-2 and k < 21:
            # Clear the aux sim
            aux_sim.initialize(tool_centers=tool_centers, tool_thetas=tool_thetas)

            # Place active particles into aux sim
            actives = np.where(part_active==1)[0]
            active_pos = part_pos[actives, :]
            active_vel = part_pos[actives, :]
            active_ages = part_ages[actives]
            aux_sim.emit_particles(len(actives), 0, active_pos, active_vel, active_ages)

            aux_sim.forward()
            loss = aux_sim.loss[None]
        
            if loss <= best_loss:
                best_loss = loss
                best_iter = k
                best_states = tool_centers.copy()

            aux_sim.backward()
            tool_centers_grads = aux_sim.tool_centers.grad.to_numpy()

            tool_centers_grads = np.clip(tool_centers_grads, -10, 10)

            tool_centers -= lr * tool_centers_grads

            k += 1

            if k % 20 == 0:
                lr *= 0.95

        # Project the solved point
        best_point = best_states[1,:]
        # actual_sim.compute_distance_project(i-1, best_point)
        # actual_sim.project_board(i-1, best_point)
        # best_point = actual_sim.projected_board.to_numpy()

        # Take the first step in the optimal trajectory will be used as init for future GD
        for j in range(0,aux_sim.max_timesteps):
            tool_centers[j,:] = best_point

    # The first step in the optimal trajectory will be taken to the actual sim
    print(best_point)
    actual_sim.take_action(i, best_point)
    final_tool_trajectory[i,:] = best_point
