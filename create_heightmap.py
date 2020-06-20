import numpy as np
from PIL import Image

x = np.arange(0,256,1)
y = np.arange(0,256,1)

x_center = 125
y_center = 125

x_sigma_sqr = 10
y_sigma_sqr = 10

A = 0.5

z = np.ones((x.shape[0], y.shape[0], 3))

# for i in range(x.shape[0]):
#     for j in range(y.shape[0]):
#         z[i,j,:] -= A*np.exp(-1 * (((x[i]-x_center)/(2*x_sigma_sqr))**2 + ((y[j]-y_center)/(2*y_sigma_sqr))**2) )

z = z*255

z[z<=0] = 0
z[z>=255] = 255

img = Image.fromarray(z.astype('uint8'), "RGB")
img.save("./test_img.png")


