import pptk
import numpy as np
import time

P = np.random.rand(10000,3)*100
v = pptk.viewer(P)
v.set(point_size=0.05)

while True:
    P = np.random.rand(10000,3)*100
    v.clear()
    v.load(P)
    time.sleep(0.1)