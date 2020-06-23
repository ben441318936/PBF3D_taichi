import json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Projection parameters 

width, height =  1938, 1095

mapMat = np.array([[0,             0,       1.08539,         -0.34732],
                   [-1.92098,      0,             0,          4.97534],
                   [0,            -1,             0,          5.64356],
                   [0,            -1,             0,            6.24]])

def mapPos(pos, mapMat, width, height):
    if pos[0] ==0 and pos[1] == 0 and pos[2] == 0 and pos[3] == 0:
        return np.full((2,1), np.nan)

    temp = mapMat @ pos

    if (temp[3] != 0):
        # convert x and y from clip space to window coordinates
        temp[0] = (temp[0] / temp[3] + 1) * .5 * width
        temp[1] = (temp[1] / temp[3] + 1) * .5 * height
    else:
        return np.full((2,1), np.nan)

    # Rounding and ensure non-negative
    results = np.array([temp[0],temp[1]])
    results = np.round(results)
    results[results <= 0] = 0
    results[results[0] >= width] = width - 1
    results[results[1] >= height] = height - 1
    
    return results

num_frames = 61

#for k in range(num_frames):
for k in range(1,61):
    s = "./PData/frame_{:05d}.json".format(k+3)

    # Read JSON
    with open(s) as f:
        data = json.load(f)

    data_list = []

    for it in data["Items"]:
        temp_list = []
        for coord in it.values():
            temp_list.append(coord)
        data_list.append(temp_list)

    # Convert to Numpy and do mapping
    data_numpy = np.array(data_list)
    num_particles = data_numpy.shape[0]
    img_numpy = np.zeros((width, height, 3))

    for i in range(num_particles):
        pix = mapPos(data_numpy[i,:], mapMat, width, height)
        if not np.isnan(pix[0]) and not np.isnan(pix[1]):
            img_numpy[int(pix[0]),int(pix[1]),:] = np.ones((3,))

    # Use Gaussian blur to smooth out and treshold
    img_numpy = gaussian_filter(img_numpy, sigma=30)
    img_numpy[img_numpy > 0.001] = 1

    img_numpy = np.swapaxes(img_numpy,0,1) * 255
    img_numpy = img_numpy[::-1, :]
    image = Image.fromarray(np.uint8(img_numpy),mode="RGB")
    image.putalpha(255)

    s = "./PImages/frame_{:05d}.png".format(k)
    image.save(s)

    img1 = Image.open("./Recordings/frames/image_{:04d}.png".format(k))
    img2 = Image.open("./PImages/frame_{:05d}.png".format(k))

    img3 = Image.blend(img1, img2, 0.3)
    img3.save("./blended/frame_{:05d}.png".format(k))





    




