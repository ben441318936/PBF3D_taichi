import pickle
import numpy as np

a = np.zeros((3,2))
b = np.ones((2,2))

print("a",a)
print("b",b)

a[2,:] = b[0,:]
b[0,:] = np.array([3,3])

print("a",a)
print("b",b)