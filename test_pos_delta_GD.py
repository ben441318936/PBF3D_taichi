from hand_grad_sim import HandGradSim
import numpy as np

sim = HandGradSim()

initial_pos0 = np.array([10.0, 10.0])
initial_vel0 = np.array([1.0, 0.0])

initial_pos1 = np.array([20.0, 10.0])
initial_vel1 = np.array([1.0, 0.0])

initial_pos2 = np.array([10.5, 10.0])
initial_vel2 = np.array([1.0, 0.0])

initial_pos3 = np.array([8, 10.0])
initial_vel3 = np.array([1.0, 0.0])

best_loss = 100
best_iter = 0

loss = 100
k = 0

lr = 1e-1

while loss > 1e-7 and k < 1000:
    print("GD iter {}".format(k))

    sim.initialize()
    sim.place_particle(0, 0, initial_pos0, initial_vel0)
    sim.place_particle(0, 1, initial_pos1, initial_vel1)
    sim.place_particle(0, 2, initial_pos2, initial_vel2)
    sim.place_particle(0, 3, initial_pos3, initial_vel3)

    sim.forward()

    print("loss:", sim.loss[None])

    for i in range(sim.num_particles):
        print("Final pos {}:".format(i), sim.positions[9,i][0], sim.positions[9,i][1])

    sim.backward()

    grads = sim.positions.grad.to_numpy()

    if k == 96:
        print("Dumping deltas")
        deltas = sim.position_deltas.to_numpy()
        print(deltas)

        print("Dumping gradients")
        print("Pos grad")
        print(grads)

        print("Pos iter grad")
        grads_iter = sim.positions_iter.grad.to_numpy()
        print(grads_iter)
        
        print("Pos delta grad")
        grads_delta = sim.position_deltas.grad.to_numpy()
        print(grads_delta)

    grads = np.clip(grads, -1e2, 1e2)

    initial_pos0 -= lr * grads[0,0]
    initial_pos1 -= lr * grads[0,1]
    initial_pos2 -= lr * grads[0,2]
    initial_pos3 -= lr * grads[0,3]
    print("New init pos 0:", initial_pos0[0], initial_pos0[1])
    print("New init pos 1:", initial_pos1[0], initial_pos1[1])
    print("New init pos 2:", initial_pos2[0], initial_pos2[1])
    print("New init pos 3:", initial_pos3[0], initial_pos3[1])

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
