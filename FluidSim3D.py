# 3D Fluid simulation using position-based fluids
# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Based on 2D implementation by Ye Kuang (k-ye)

import taichi as ti

@ti.data_oriented
class FluidSim3D(object):
    def __init__(self):
        return