import numpy as np
from matplotlib import pyplot as plt

'''
1D example
'''

# x = np.arange(-10, 10, 0.1)
# f = x**2

# grad = np.gradient(f, x)

# plt.plot(x, f)
# plt.plot(x, grad)
# plt.legend(["x^2", "gradient"])
# plt.show()

'''
2D example
'''

# x = np.arange(-2, 4, 0.5)
# x_var = 1

# y = np.arange(-3, 3, 0.1)
# y_var = 1

# X, Y = np.meshgrid(x, y)

# Z = ((2*np.pi)**2 * x_var * y_var)**(-1/2) * np.exp(-1/2 * ((X-0.5)**2/x_var + Y**2/y_var) ) * 0.5

# grad = np.gradient(Z, y, x)

# plt.quiver(X, Y, grad[1], grad[0])

# plot_x = X.reshape((np.prod(X.shape),))
# plot_y = Y.reshape((np.prod(Y.shape),))
# plot_z = Z.reshape((np.prod(Z.shape),))

# plt.scatter(plot_x, plot_y, c=plot_z, cmap="coolwarm", s=2)

# plt.show()