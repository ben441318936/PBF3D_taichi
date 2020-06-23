import numpy as np

def mapPos(pos, mapMat, width, height):
    temp = mapMat @ pos

    if (temp[3] != 0):
        # convert x and y from clip space to window coordinates
        temp[0] = (temp[0] / temp[3] + 1) * .5 * width
        temp[1] = (temp[1] / temp[3] + 1) * .5 * height
    else:
        temp = np.zeros((3,))
    
    return np.array([temp[0],temp[1]])


mapMat = np.array([[0,             0,       1.08539,         -0.34732],
                   [-1.92098,      0,             0,          4.97534],
                   [0,            -1,             0,          5.64356],
                   [0,            -1,             0,            6.24]])

print(mapMat)

#pos = np.array([3.927856, 3.67, -2.047823, 1])
pos = np.array([1.252144, 3.67, -2.047823, 1])
print(pos)

pixLoc = mapPos(pos, mapMat, 1938, 1095)
print(pixLoc)