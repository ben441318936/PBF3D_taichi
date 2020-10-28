import numpy as np

# key_points = [np.array([2,1,7]), 
#               np.array([7,1,7]), 
#               np.array([2,1,7]), 
#               np.array([2,1,2])]

# steps = np.arange(2, 7+0.05, 0.05)

# trajectory = []

# for i in range(steps.shape[0]):
#     trajectory.append(np.array([steps[i], 1, 7]))
# for i in reversed(range(steps.shape[0])):
#     trajectory.append(np.array([7, 1, steps[i]]))
# for i in reversed(range(steps.shape[0])):
#     trajectory.append(np.array([steps[i], 1, 2]))

# trajectory = np.array(trajectory)
# print(trajectory.shape)

# np.save("./cavity1_manual_path.npy", trajectory)

steps1 = np.arange(2, 7+0.05, 0.05)
steps2 = np.arange(4, 7+0.05, 0.05)

trajectory = []

for i in range(steps1.shape[0]):
    trajectory.append(np.array([steps1[i], 1, 7]))
for i in reversed(range(steps2.shape[0])):
    trajectory.append(np.array([7, 1, steps2[i]]))
for i in range(300 - steps1.shape[0] - steps2.shape[0]):
    trajectory.append(trajectory[-1])

trajectory = np.array(trajectory)
print(trajectory.shape)

np.save("./cavity2_manual_path_left.npy", trajectory)

steps1 = np.arange(2, 7+0.05, 0.05)
steps2 = np.arange(2, 4+0.05, 0.05)

trajectory = []

for i in range(steps1.shape[0]):
    trajectory.append(np.array([steps1[i], 1, 2]))
for i in range(steps2.shape[0]):
    trajectory.append(np.array([7, 1, steps2[i]]))
for i in range(300 - steps1.shape[0] - steps2.shape[0]):
    trajectory.append(trajectory[-1])

trajectory = np.array(trajectory)
print(trajectory.shape)

np.save("./cavity2_manual_path_right.npy", trajectory)
