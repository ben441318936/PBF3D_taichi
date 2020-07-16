import numpy as np

def translate(p, c):
    return p + c

def rotate(p, theta):
    new_p = np.zeros((2,))
    new_p[0] = p[0] * np.cos(theta) - p[1] * np.sin(theta)
    new_p[1] = p[0] * np.sin(theta) + p[1] * np.cos(theta)
    return new_p

center = np.array([10.0, 10.0])
theta = np.pi/4
dims = np.array([1.0, 10.0])

left = center[0] - dims[0]/2
right = center[0] + dims[0]/2
bot = center[1] - dims[1]/2
top = center[1] + dims[1]/2

p = np.array([10.0 - dims[0]/2, 12.0])
print(p)

p = translate(p, -1 * center)
p = rotate(p, -1 * theta)
p = translate(p, center)
print(p)