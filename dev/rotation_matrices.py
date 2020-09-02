import numpy as np

def rot_x(theta):
    return np.array([[1,               0,              0],
                     [0,               np.cos(theta),  -np.sin(theta)],
                     [0,               np.sin(theta),  np.cos(theta)]])

def rot_y(theta):
    return np.array([[np.cos(theta),   0,              np.sin(theta)],
                     [0,               1,              0],
                     [-np.sin(theta),  0,              np.cos(theta)]])

def rot_z(theta):
    return np.array([[np.cos(theta),   -np.sin(theta), 0],
                     [np.sin(theta),   np.cos(theta),  0],
                     [0,               0,              1]])

a = rot_x(-np.pi/2)
b = rot_y(np.pi*(225/180))
c = rot_z(np.pi*(10/180))

d = c @ b @ a
print(d)