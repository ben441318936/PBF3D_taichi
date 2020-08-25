from hand_grad_sim_3D import HandGradSim3D
import numpy as np

time = 10

sim = HandGradSim3D(max_timesteps=time, num_particles=5, do_save_npy=True, do_emit=False)

start = 0.1
spacing = 0.1

initial_pos = np.array([[0.1, 1, start + 0 * spacing],
                        [0.1, 1, start + 1 * spacing],
                        [0.1, 1, start + 3 * spacing],
                        [0.1, 1, start + 4 * spacing]], dtype=np.float)

initial_vel = np.array([[10, 0, 5],
                        [10, 0, 5],
                        [10, 0, 5],
                        [10, 0, 5]])

lr = 1e-1
loss = 100
k = 0

while loss > 1e-2 and k < 1:
    print("Iteration", k)

    sim.initialize()
    sim.emit_particles(4, 0, initial_pos, initial_vel)

    sim.forward()
    print("Loss:", sim.loss[None])
    positions = sim.positions.to_numpy()
    print("Final pos")
    print(positions[time-1,:,:])

    sim.backward()

    pos_iter_grads = sim.positions_iter.grad.to_numpy()
    delta_grads = sim.position_deltas.grad.to_numpy()
    position_grads = sim.positions.grad.to_numpy()

    for j in reversed(range(0,time)):
        print("Frame", j)
        print("Pos grads")
        print(position_grads[j,:,:])

        for i in reversed(range(0,3)):
            print("Iter", i)

            print("Pos iter grads")
            print(pos_iter_grads[j,i,:])
            
            print("Delta grads")
            print(delta_grads[j,i,:])
        
    
    print("Pos grads")
    print(position_grads[0,:,:])

    position_grads = np.clip(position_grads, -10, 10)

    # velocity_grads = sim.velocities.grad.to_numpy()
    # print(velocity_grads[0,:,:])

    initial_pos -= lr * position_grads[0,0:4,:]

    k += 1

