import numpy as np

a = np.eye(4)
a[:3,3] = np.array([5,25,-5])

b = np.eye(4)
b[:3,3] = np.array([120,10,100])

c = a @ b
print(c)