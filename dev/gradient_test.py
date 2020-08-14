from hand_grad_sim_3D import HandGradSim3D
import numpy as np

time = 4

sim = HandGradSim3D(max_timesteps=time, num_particles=5, do_save_npy=True, do_emit=False)

start = 9
spacing = 0.5

initial_pos = np.array([[10, 10, start],
                        [10, 10, start + spacing],
                        [10, 10, start + 2*spacing],
                        [10, 10, start + 3*spacing]])

initial_vel = np.array([[10, 0, 5],
                        [10, 0, 5],
                        [10, 0, 5],
                        [10, 0, 5]])

lr = 1e-1
loss = 100
k = 0

while loss > 1e-2 and k < 30:
    print("Iteration", k)

    sim.initialize()
    sim.emit_particles(4, 0, initial_pos, initial_vel)

    sim.forward()
    print(sim.loss[None])
    positions = sim.positions.to_numpy()
    print(positions[time-1,:,:])

    sim.backward()

    position_grads = sim.positions.grad.to_numpy()
    print(position_grads[0,:,:])

    position_grads = np.clip(position_grads, -10, 10)

    # velocity_grads = sim.velocities.grad.to_numpy()
    # print(velocity_grads[0,:,:])

    initial_pos -= lr * position_grads[0,0:4,:]

    k += 1

