import numpy as np

one_x = 9.5 - 0.2
two_z = 0.5 + 0.2
three_x = 0.5 + 0.2
four_z = 2.9 - 0.2
five_x = 6.5 + 0.2
six_z = 5.3 + 0.2
seven_x = 0.5 + 0.2
eight_z = 8.2 - 0.5 - 0.2

emit_pos = []
emit_vel = []

z = np.linspace(0.5, 7.5-1, 7)
print(z)
print(len(emit_pos))
for p in z:
    emit_pos.append(np.array([[one_x, 1, p], [one_x, 1, p+0.5], [one_x, 1, p+1]]))
    emit_vel.append(np.array([[-5, 0, 0], [-5, 0, 0], [-5, 0, 0]]))

x = np.linspace(0.5, 9.5-1, 9)
print(x)
print(len(emit_pos))
for p in x:
    emit_pos.append(np.array([[p, 1, two_z], [p+0.5, 1, two_z], [p+1, 1, two_z]]))
    emit_vel.append(np.array([[0, 0, 5], [0, 0, 5], [0, 0, 5]]))

z = np.linspace(0.5, 2.9-0.8, 3)
print(z)
print(len(emit_pos))
for p in z:
    emit_pos.append(np.array([[three_x, 1, p], [three_x, 1, p+0.4], [three_x, 1, p+0.8]]))
    emit_vel.append(np.array([[5, 0, 0], [5, 0, 0], [5, 0, 0]]))

x = np.linspace(0.5, 6.5-1, 6)
print(x)
print(len(emit_pos))
for p in x:
    emit_pos.append(np.array([[p, 1, four_z], [p+0.5, 1, four_z], [p+1, 1, four_z]]))
    emit_vel.append(np.array([[0, 0, -5], [0, 0, -5], [0, 0, -5]]))

z = np.linspace(2.9, 5.3-0.8, 3)
print(z)
print(len(emit_pos))
for p in z:
    emit_pos.append(np.array([[five_x, 1, p], [five_x, 1, p+0.4], [five_x, 1, p+0.8]]))
    emit_vel.append(np.array([[5, 0, 0], [5, 0, 0], [5, 0, 0]]))

x = np.linspace(0.5, 6.5-1, 6)
print(x)
print(len(emit_pos))
for p in x:
    emit_pos.append(np.array([[p, 1, six_z], [p+0.5, 1, six_z], [p+1, 1, six_z]]))
    emit_vel.append(np.array([[0, 0, 5], [0, 0, 5], [0, 0, 5]]))

z = np.linspace(5.3, 7.7-0.8, 3)
print(z)
print(len(emit_pos))
for p in z:
    emit_pos.append(np.array([[seven_x, 1, p], [seven_x, 1, p+0.4], [seven_x, 1, p+0.8]]))
    emit_vel.append(np.array([[5, 0, 0], [5, 0, 0], [5, 0, 0]]))

x = np.linspace(0.5, 9.5-1, 9)
print(x)
print(len(emit_pos))
for p in x:
    emit_pos.append(np.array([[p, 1, eight_z], [p+0.5, 1, eight_z], [p+1, 1, eight_z]]))
    emit_vel.append(np.array([[0, 0, -5], [0, 0, -5], [0, 0, -5]]))

emit_pos = np.array(emit_pos)
emit_vel = np.array(emit_vel)

print(emit_pos.shape[0])

np.save("./cavity1_emit_pos.npy", emit_pos)
np.save("./cavity1_emit_vel.npy", emit_vel)