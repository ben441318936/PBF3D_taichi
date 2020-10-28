import numpy as np

# First compute trajectory length
exp_num = 520
time = 500

trajectory_lengths = []

for i in range(46):
    dist = 0
    path_prefix = "/home/jingbin/Documents/Github/PBF3D_taichi/viz_results/3D/new_MPC/exp{}/tool/".format(exp_num+i)
    for j in range(200, time-1):
        tool1 = np.load(path_prefix+"frame_{}.npy".format(j))
        tool1[1] = 0
        tool2 = np.load(path_prefix+"frame_{}.npy".format(j+1))
        tool2[1] = 0
        dist += np.linalg.norm(tool2 - tool1)
    trajectory_lengths.append(dist)
trajectory_lengths = np.array(trajectory_lengths)

# Get emission points and plot
emit_pos = np.load("cavity1_emit_pos.npy")

new_emit_pos = np.zeros((emit_pos.shape[0],2))

new_emit_pos[:,0] = emit_pos[:,1,0]
new_emit_pos[:,1] = emit_pos[:,1,2]

import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.scatter(-new_emit_pos[:,1], -new_emit_pos[:,0], c=trajectory_lengths, cmap="jet")
plt.colorbar()

l = trajectory_lengths.tolist()

for i, n in enumerate(l):
    txt = "{:.2f}".format(n)
    plt.annotate(txt, (-new_emit_pos[i,1], -new_emit_pos[i,0]))

plt.show()