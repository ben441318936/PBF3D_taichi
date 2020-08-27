import numpy as np

def find_nearest_ind(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_inds(x, y, z, p):
    inds = [0, 0, 0]
    inds[0] = find_nearest_ind(x,p[0])
    inds[1] = find_nearest_ind(y,p[1])
    inds[2] = find_nearest_ind(z,p[2])
    return inds


x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
z = np.arange(-5, 5, 0.1)

p = [2, 2, 2]

inds = get_inds(x, y, z, p)

print(inds)