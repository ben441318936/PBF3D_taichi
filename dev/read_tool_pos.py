import numpy as np

exp = "exp700"
tool_path_prefix = "/home/jingbin/Documents/Github/PBF3D_taichi/viz_results/3D/new_MPC/{}/tool/".format(exp)

num_steps = 400

trajectory = []

for i in range(100,num_steps):
    if i % 5 == 0:
        path = tool_path_prefix+"frame_{}.npy".format(i)
        tool = np.load(path)
        trajectory.append(tool)

trajectory = np.array(trajectory)
print(trajectory.shape)
# print(trajectory)

np.save(tool_path_prefix+"trajectory.npy", trajectory)

# traj_path = "./cavity2_manual_path_left.npy"
# traj = np.load(traj_path)

# trajectory = []

# for i in range(0, 300):
#     if i % 5 == 0:
#         # trajectory.append(traj[i])
#         # trajectory.append(np.array([1,1,1])) # case 1 fixed eimssion
#         # trajectory.append(np.array([1,1,7])) # case 1 fixed end
#         # trajectory.append(np.array([7,1,4.5])) # case 1 fixed middle
#         # trajectory.append(np.array([7,1,4])) # case 2 fixed emission
#         # trajectory.append(np.array([1,1,7])) # case 2 fixed end left
#         trajectory.append(np.array([1,1,1])) # case 2 fixed end right

# trajectory = np.array(trajectory)
# print(trajectory.shape)

# np.save("./case2_fixed_end_right.npy", trajectory)
