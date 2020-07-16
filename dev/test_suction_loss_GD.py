from hand_grad_sim import HandGradSim
import numpy as np
import pickle

iter_states = {}

sim = HandGradSim()

best_states = np.zeros((sim.max_timesteps,sim.dim))

board_states = np.zeros((sim.max_timesteps,sim.dim))
for i in range(sim.max_timesteps):
    board_states[i,:] = np.array([10.0, 20.0])

init_pos_path = "./states/init_pos.obj"
with open(init_pos_path, "rb") as f:
    init_pos = pickle.load(f)

init_vel_path = "./states/init_vel.obj"
with open(init_vel_path, "rb") as f:
    init_vel = pickle.load(f)  


best_loss = 1e5
best_iter = 0

loss = best_loss
k = 0

lr = 1e1

while loss > 1e-2 and k < 101:
    print("GD iter {}".format(k))

    sim.initialize(board_states)

    sim.emit_particles(100, 0, init_pos, init_vel)

    sim.forward()

    print("loss:", sim.loss[None])

    loss = sim.loss[None]
    
    if loss <= best_loss:
        best_loss = loss
        best_iter = k
        best_states = board_states.copy()

    if k % 10 == 0:
        iter_states["iter{}".format(k)] = board_states.copy()
        
    sim.backward()

    board_grads = sim.board_states.grad.to_numpy()

    # for i in range(sim.max_timesteps):
    #     print("Board grads {}:".format(i), board_grads[i])

    board_grads = np.clip(board_grads, -10, 10)

    if k == 100:
        print(board_grads[-1])

    board_states -= lr * board_grads

    k += 1

    if k % 100 == 0:
        lr *= 0.95


print("Best loss is {} at iter {}".format(best_loss, best_iter))

iter_states_path = "./states/set2/iter_states_1.obj"
with open(iter_states_path, "w+b") as f:
    pickle.dump(iter_states, f)

best_states_path = "./states/set2/best_states_1.obj"
with open(best_states_path, "w+b") as f:
    pickle.dump(best_states, f)


