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

print("Gravity forward")
sim.gravity_forward(1)

print("intermediate pos 0:", sim.positions_intermediate[1,0][0], sim.positions_intermediate[1,0][1])
    
sim.clear_neighbor_info()
sim.update_grid(1)
sim.find_particle_neighbors(1)

print("Compute lambdas")
sim.compute_lambdas_forward(1)
print("lambda 0:", sim.lambdas[0])
print("lambda 1:", sim.lambdas[1])

print("Compute position deltas")
sim.compute_position_deltas_forward(1)
print("delta 0:", sim.position_deltas[0][0], sim.position_deltas[0][1])
print("delta 1:", sim.position_deltas[1][0], sim.position_deltas[1][1])
print("delta 2:", sim.position_deltas[2][0], sim.position_deltas[2][1])
print("delta 3:", sim.position_deltas[3][0], sim.position_deltas[3][1])

print("Apply position deltas")
sim.apply_position_deltas_forward(1)
print("pos 0:", sim.positions[1,0][0], sim.positions[1,0][1])
print("pos 1:", sim.positions[1,1][0], sim.positions[1,1][1])
print("pos 2:", sim.positions[1,2][0], sim.positions[1,2][1])
print("pos 3:", sim.positions[1,3][0], sim.positions[1,3][1])

sim.compute_loss_forward()
print("loss:", sim.loss[None])

sim.clear_grads()

print("Loss backward")
sim.compute_loss_backward()
print("grad to pos 0:", sim.positions.grad[1,0][0], sim.positions.grad[1,0][1])
print("grad to pos 1:", sim.positions.grad[1,1][0], sim.positions.grad[1,1][1])
print("grad to pos 2:", sim.positions.grad[1,2][0], sim.positions.grad[1,2][1])
print("grad to pos 3:", sim.positions.grad[1,3][0], sim.positions.grad[1,3][1])

print("Apply position deltas backward")
sim.apply_position_deltas_backward(1)

print("Compute position deltas backward")
sim.compute_position_deltas_backward(1)

print("grad to inter pos 0:", sim.positions_intermediate.grad[1,0][0], sim.positions_intermediate.grad[1,0][1])
print("grad to inter pos 1:", sim.positions_intermediate.grad[1,1][0], sim.positions_intermediate.grad[1,1][1])
print("grad to inter pos 2:", sim.positions_intermediate.grad[1,2][0], sim.positions_intermediate.grad[1,2][1])
print("grad to inter pos 3:", sim.positions_intermediate.grad[1,3][0], sim.positions_intermediate.grad[1,3][1])

print("grad to lambda 0:", sim.lambdas.grad[0])
print("grad to lambda 1:", sim.lambdas.grad[1])
print("grad to lambda 2:", sim.lambdas.grad[2])
print("grad to lambda 3:", sim.lambdas.grad[3])

print("Compute lambdas backward")
sim.compute_lambdas_backward(1)
print("grad to inter pos 0:", sim.positions_intermediate.grad[1,0][0], sim.positions_intermediate.grad[1,0][1])
print("grad to inter pos 1:", sim.positions_intermediate.grad[1,1][0], sim.positions_intermediate.grad[1,1][1])
print("grad to inter pos 2:", sim.positions_intermediate.grad[1,2][0], sim.positions_intermediate.grad[1,2][1])
print("grad to inter pos 3:", sim.positions_intermediate.grad[1,3][0], sim.positions_intermediate.grad[1,3][1])

print("Gravity backward")
sim.gravity_backward(1)
print("grad to pos 0:", sim.positions.grad[0,0][0], sim.positions.grad[0,0][1])
print("grad to pos 1:", sim.positions.grad[0,1][0], sim.positions.grad[0,1][1])
print("grad to pos 2:", sim.positions.grad[0,2][0], sim.positions.grad[0,2][1])
print("grad to pos 3:", sim.positions.grad[0,3][0], sim.positions.grad[0,3][1])