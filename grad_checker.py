import taichi as ti
import numpy as np
from hand_grad_sim import HandGradSim

sim = HandGradSim(max_timesteps=10, num_particles=10, do_render=False, do_emit=False)
eps = 1e-3

def check_gravity():
    # Set up
    board_states = np.zeros((10,2))
    for i in range(sim.max_timesteps):
        board_states[i,:] = np.array([10.0, 20.0])
    sim.initialize(board_states)

    # Get analytical grad
    init_pos = np.random.uniform(0, 40, (10,2))
    init_vel = np.random.uniform(-10, 10, (10,2))
    sim.emit_particles(10, 0, init_pos, init_vel)

    init_pos = sim.positions.to_numpy()[0,:,:]
    init_vel = sim.velocities.to_numpy()[0,:,:]

    sim.clear_neighbor_info()
    sim.gravity_forward(1)
    # Grads are computed
    sim.clear_global_grads()
    for i in range(sim.num_particles):
        if sim.particle_active[0,i] == 1:
            sim.positions_iter.grad[1,0,i][0] = 1
            sim.positions_iter.grad[1,0,i][1] = 1
    sim.gravity_backward(1)

    pos_grads = sim.positions.grad.to_numpy()[0,:,:]
    vel_grads = sim.velocities.grad.to_numpy()[0,:,:]

    # print("Analytical grads")
    # print("Positions:", pos_grads)
    # print("Velocities:", vel_grads)

    tot_rel_error = 0
    runs = 100
    counts = 0
    # Get numerical grads
    for j in range(runs):
        pos_offset = np.random.uniform(-1, 1, (10,2))
        vel_offset = np.random.uniform(-1, 1, (10,2))

        sim.initialize(board_states)
        sim.emit_particles(10, 0, init_pos + eps*pos_offset, init_vel + eps*vel_offset)
        sim.clear_neighbor_info()
        sim.gravity_forward(1)

        output1 = sim.positions_iter.to_numpy()[1,0,:,:]
        output1 = np.sum(output1)

        sim.initialize(board_states)
        sim.emit_particles(10, 0, init_pos - eps*pos_offset, init_vel - eps*vel_offset)
        sim.clear_neighbor_info()
        sim.gravity_forward(1)

        output2 = sim.positions_iter.to_numpy()[1,0,:,:]
        output2 = np.sum(output2)

        numerical_dif = 1/(2*eps) * (output1 - output2)
        analytical_dif = np.sum(pos_grads * pos_offset + vel_grads * vel_offset)

        if numerical_dif == 0:
            pass
        else:
            # print("Numerical dif is", numerical_dif)
            # print("Analytical dif is", analytical_dif)
            relative_error = np.abs(numerical_dif - analytical_dif) / np.abs(numerical_dif)
            # print("Relative error is", relative_error)
            tot_rel_error += relative_error
            counts += 1

    print("Avg relative error", tot_rel_error/counts)
    return tot_rel_error / counts


tot_error = 0
for k in range(10):
    tot_error += check_gravity()
print("Final avg relative error", tot_error/10)



