from hand_grad_sim import HandGradSim
import numpy as np

sim = HandGradSim()

loss = 10
k = 0
target_lambda = 0.03

lr = 10

initial_pos0 = np.array([10.0, 10.0])
initial_vel0 = np.array([10.0, 0.0])

initial_pos1 = np.array([20.0, 10.0])
initial_vel1 = np.array([10.0, 0.0])

initial_pos2 = np.array([10.5, 10.0])
initial_vel2 = np.array([10.0, 0.0])

initial_pos3 = np.array([9.5, 10.0])
initial_vel3 = np.array([10.0, 0.0])

while loss > 1e-10 and k < 100:

    sim.initialize()
    sim.place_particle(0, 0, initial_pos0, initial_vel0)
    sim.place_particle(0, 1, initial_pos1, initial_vel1)
    sim.place_particle(0, 2, initial_pos2, initial_vel2)
    sim.place_particle(0, 3, initial_pos3, initial_vel3)

    sim.gravity_forward(1)
    
    sim.clear_neighbor_info()
    sim.update_grid(1)
    sim.find_particle_neighbors(1)
    sim.compute_lambdas_forward(1)

    sim.clear_grads()
    sim.compute_lambdas_backward(1)
    sim.gravity_backward(1)

    s = 0
    for i in range(sim.num_particles):
        s += sim.lambdas[i]

    print("Sum lambdas:", s)
    loss =  1/2 * (s - target_lambda)**2
    print("Loss:", loss)
    print("Grad lambda to particle 0 intermediate:", sim.positions_intermediate.grad[1,0][0], sim.positions_intermediate.grad[1,0][1])
    print("Grad lambda to particle 0 position initial:", sim.positions.grad[0,0][0], sim.positions.grad[0,0][1])

    loss_grad = (s - target_lambda)
        
    # initial_pos0 -= lr * loss_grad * np.array([sim.positions.grad[0,0][0], sim.positions.grad[0,0][1]])
    # initial_pos1 -= lr * loss_grad * np.array([sim.positions.grad[0,1][0], sim.positions.grad[0,1][1]])
    initial_pos2 -= lr * loss_grad * np.array([sim.positions.grad[0,2][0], sim.positions.grad[0,2][1]])
    # initial_pos3 -= lr * loss_grad * np.array([sim.positions.grad[0,3][0], sim.positions.grad[0,3][1]])

    print("New initial pos 0:", initial_pos0[0], initial_pos0[1])
    print("New initial pos 1:", initial_pos1[0], initial_pos1[1])
    print("New initial pos 2:", initial_pos2[0], initial_pos2[1])
    print("New initial pos 3:", initial_pos3[0], initial_pos3[1])

    k += 1






