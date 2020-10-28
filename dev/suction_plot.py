import numpy as np
# from scipy.io import savemat
import matplotlib.pyplot as plt

'''
exps = [500, 511, 512, 513, 514]

time = 500

exp_nums = []

for e in exps:
    exp = "exp{}".format(e)
    particle_path_prefix = "/home/jingbin/Documents/Github/PBF3D_taichi/viz_results/3D/new_MPC/{}/particles/".format(exp)   
    nums = []
    for i in range(time):
        particle_path = particle_path_prefix+"frame_{}.npy".format(i)
        particles = np.load(particle_path)
        nums.append(particles.shape[0])
    exp_nums.append(nums)

# Final count
exp_nums = np.array(exp_nums)
exp_nums = exp_nums[:,200:]
print(exp_nums[:,-1])

# AUC = np.sum(exp_nums, axis=1) * 1/100
# print(AUC)

t = np.arange(200,time)

# Convergence time
tau_50 = []
for i in range(0,len(exps)):
    # dif = exp_nums[i,0] - exp_nums[i,-1]
    target = exp_nums[i,0] - 0.5 * exp_nums[i,0]
    for j in range(0, len(t)):
        if exp_nums[i,j] <= target:
            tau_50.append(j)
            break
tau_50 = np.array(tau_50)
print("50% tau", tau_50)

tau_90 = []
for i in range(0,len(exps)):
    # dif = exp_nums[i,0] - exp_nums[i,-1]
    target = exp_nums[i,0] - 0.9 * exp_nums[i,0]
    for j in range(0, len(t)):
        if exp_nums[i,j] <= target:
            tau_90.append(j)
            break
tau_90 = np.array(tau_90)
print("90% tau", tau_90)

plt.figure(figsize=(7,5))

for i in range(0,len(exps)):
    z = 0
    if i == 0:
        z = 10
    plt.plot(t, exp_nums[i,:], zorder=z)

plt.legend(["Our method",
            "Fixed: at emission point", 
            "Fixed: at end of path", 
            "Fixed: at middle of path",
            "End-to-emit trajectory"])
plt.xlabel("Time steps")
plt.ylabel("Number of remaining particles")
# plt.show()
plt.savefig("./suction_curve_cavity1.png", dpi=400, bbox_inches="tight")
'''


exps = [600, 611, 612, 613, 614, 615]

time = 500

exp_nums = []

for e in exps:
    exp = "exp{}".format(e)
    particle_path_prefix = "/home/jingbin/Documents/Github/PBF3D_taichi/viz_results/3D/new_MPC/{}/particles/".format(exp)   
    nums = []
    for i in range(time):
        particle_path = particle_path_prefix+"frame_{}.npy".format(i)
        particles = np.load(particle_path)
        nums.append(particles.shape[0])
    exp_nums.append(nums)

exp_nums = np.array(exp_nums)
exp_nums = exp_nums[:,200:]
print(exp_nums[:,-1])

AUC = np.sum(exp_nums, axis=1) * 1/100
print(AUC)

t = np.arange(200,time)

# Convergence time
tau_50 = []
for i in range(0,len(exps)):
    # dif = exp_nums[i,0] - exp_nums[i,-1]
    target = exp_nums[i,0] - 0.5 * exp_nums[i,0]
    for j in range(0, len(t)):
        if exp_nums[i,j] <= target:
            tau_50.append(j)
            break
tau_50 = np.array(tau_50)
print("50% tau", tau_50)

tau_90 = []
for i in range(0,len(exps)):
    dif = exp_nums[i,0] - exp_nums[i,-1]
    target = exp_nums[i,0] - 0.9 * exp_nums[i,0]
    for j in range(0, len(t)):
        if exp_nums[i,j] <= target:
            tau_90.append(j)
            break
tau_90 = np.array(tau_90)
print("90% tau", tau_90)

plt.figure(figsize=(7,5))

for i in range(0,len(exps)):
    z = 0
    if i == 0:
        z = 10
    plt.plot(t, exp_nums[i,:], zorder=z)

plt.legend(["Our method",
            "Fixed: at emission point",
            "Fixed: at left end of path", 
            "Fixed: at right end of path", 
            "End-to-emit trajectory from left",
            "End-to-emit trajectory from right"])
plt.xlabel("Time steps")
plt.ylabel("Number of remaining particles")
# plt.show()
plt.savefig("./suction_curve_cavity2.png", dpi=400, bbox_inches="tight")


