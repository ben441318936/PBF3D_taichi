from hand_grad_sim import HandGradSim
import numpy as np

sim = HandGradSim()

initial_pos0 = np.array([10.0, 10.0])
initial_vel0 = np.array([10.0, 0.0])

initial_pos1 = np.array([20.0, 10.0])
initial_vel1 = np.array([10.0, 0.0])

initial_pos2 = np.array([10.1, 10.0])
initial_vel2 = np.array([10.0, 0.0])

initial_pos3 = np.array([9.8, 10.0])
initial_vel3 = np.array([10.0, 0.0])

print("Init")

sim.initialize()
sim.place_particle(0, 0, initial_pos0, initial_vel0)
sim.place_particle(0, 1, initial_pos1, initial_vel1)
sim.place_particle(0, 2, initial_pos2, initial_vel2)
sim.place_particle(0, 3, initial_pos3, initial_vel3)

print("init pos 0:", sim.positions[0,0][0], sim.positions[0,0][1])

sim.forward()

sim.compute_loss_forward()
print("loss:", sim.loss[None])

sim.clear_global_grads()

print("Loss backward")
sim.compute_loss_backward()
print("grad to pos 0:", sim.positions.grad[2,0][0], sim.positions.grad[2,0][1])
print("grad to pos 1:", sim.positions.grad[2,1][0], sim.positions.grad[2,1][1])
print("grad to pos 2:", sim.positions.grad[2,2][0], sim.positions.grad[2,2][1])
print("grad to pos 3:", sim.positions.grad[2,3][0], sim.positions.grad[2,3][1])

for i in reversed(range(1,3)):

    sim.clear_local_grads()

    print("Apply vel update backward")
    sim.update_velocity_backward(i)

    print("grad to pos 0:", sim.positions.grad[i,0][0], sim.positions.grad[i,0][1])
    print("grad to pos 1:", sim.positions.grad[i,1][0], sim.positions.grad[i,1][1])
    print("grad to pos 2:", sim.positions.grad[i,2][0], sim.positions.grad[i,2][1])
    print("grad to pos 3:", sim.positions.grad[i,3][0], sim.positions.grad[i,3][1])

    print("grad to prev 0:", sim.positions.grad[i-1,0][0], sim.positions.grad[i-1,0][1])
    print("grad to prev 1:", sim.positions.grad[i-1,1][0], sim.positions.grad[i-1,1][1])
    print("grad to prev 2:", sim.positions.grad[i-1,2][0], sim.positions.grad[i-1,2][1])
    print("grad to prev 3:", sim.positions.grad[i-1,3][0], sim.positions.grad[i-1,3][1])

    print("Apply position deltas backward")
    sim.apply_position_deltas_backward(i)

    print("Compute position deltas backward")
    sim.compute_position_deltas_backward(i)

    print("grad to inter pos 0:", sim.positions_intermediate.grad[i,0][0], sim.positions_intermediate.grad[i,0][1])
    print("grad to inter pos 1:", sim.positions_intermediate.grad[i,1][0], sim.positions_intermediate.grad[i,1][1])
    print("grad to inter pos 2:", sim.positions_intermediate.grad[i,2][0], sim.positions_intermediate.grad[i,2][1])
    print("grad to inter pos 3:", sim.positions_intermediate.grad[i,3][0], sim.positions_intermediate.grad[i,3][1])

    print("grad to lambda 0:", sim.lambdas.grad[i,0])
    print("grad to lambda 1:", sim.lambdas.grad[i,1])
    print("grad to lambda 2:", sim.lambdas.grad[i,2])
    print("grad to lambda 3:", sim.lambdas.grad[i,3])

    print("Compute lambdas backward")
    sim.compute_lambdas_backward(i)
    print("grad to inter pos 0:", sim.positions_intermediate.grad[i,0][0], sim.positions_intermediate.grad[i,0][1])
    print("grad to inter pos 1:", sim.positions_intermediate.grad[i,1][0], sim.positions_intermediate.grad[i,1][1])
    print("grad to inter pos 2:", sim.positions_intermediate.grad[i,2][0], sim.positions_intermediate.grad[i,2][1])
    print("grad to inter pos 3:", sim.positions_intermediate.grad[i,3][0], sim.positions_intermediate.grad[i,3][1])

    print("Gravity backward")
    sim.gravity_backward(i)
    print("grad to pos 0:", sim.positions.grad[i-1,0][0], sim.positions.grad[i-1,0][1])
    print("grad to pos 1:", sim.positions.grad[i-1,1][0], sim.positions.grad[i-1,1][1])
    print("grad to pos 2:", sim.positions.grad[i-1,2][0], sim.positions.grad[i-1,2][1])
    print("grad to pos 3:", sim.positions.grad[i-1,3][0], sim.positions.grad[i-1,3][1])