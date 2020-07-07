from hand_grad_sim import HandGradSim
import numpy as np

sim = HandGradSim()

initial_positions = np.array([[10.0, 10.0],
                              [15.0, 10.0],
                              [10.5, 10.0],
                              [8.0, 10.0]])
initial_velocities = np.array([[1.0, 0.0],
                               [1.0, 0.0],
                               [1.0, 0.0],
                               [1.0, 0.0]])

best_loss = 100
best_iter = 0

loss = 100
k = 0

lr = 1e-1

while loss > 1e-7 and k < 100:
    print("GD iter {}".format(k))

    sim.initialize()
    sim.emit_particles(4, 0, initial_positions[0:4,:], initial_velocities[0:4,:])

    sim.forward()

    print("loss:", sim.loss[None])

    for i in range(sim.num_particles):
        print("Final pos {}:".format(i), sim.positions[9,i][0], sim.positions[9,i][1])

    sim.backward()

    grads = sim.positions.grad.to_numpy()

    # if k == 96:
    #     print("Dumping deltas")
    #     deltas = sim.position_deltas.to_numpy()
    #     print(deltas)

    #     print("Dumping gradients")
    #     print("Pos grad")
    #     print(grads)

    #     print("Pos iter grad")
    #     grads_iter = sim.positions_iter.grad.to_numpy()
    #     print(grads_iter)
        
    #     print("Pos delta grad")
    #     grads_delta = sim.position_deltas.grad.to_numpy()
    #     print(grads_delta)

    grads = np.clip(grads, -1, 1)

    initial_positions -= lr * grads[0,:,:]


    # grads = sim.velocities.grad.to_numpy()
    # grads = np.clip(grads, -1e2, 1e2)

    # initial_vel0 -= lr * grads[0,0]
    # initial_vel1 -= lr * grads[0,1]
    # initial_vel2 -= lr * grads[0,2]
    # initial_vel3 -= lr * grads[0,3]
    # print("New init vel 0:", initial_vel0[0], initial_vel0[1])
    # print("New init vel 1:", initial_vel1[0], initial_vel1[1])
    # print("New init vel 2:", initial_vel2[0], initial_vel2[1])
    # print("New init vel 3:", initial_vel3[0], initial_vel3[1])

    loss = sim.loss[None]
    
    if loss <= best_loss:
        best_loss = loss
        best_iter = k

    k += 1

    if k % 100 == 0:
        lr *= 0.95


print("Best loss is {} at iter {}".format(best_loss, best_iter))
