import cv2
from cv2 import aruco
import numpy as np

def get_ARUCO_tvec_rmat(img):
    image = cv2.imread("aruco_test_image.jpg")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    print(ids)

    mtx = np.array([[8.90712747e2, 0.0, 4.41241852e2],
                    [0.0, 8.90712747e2, 191.90303209],
                    [0.0, 0.0, 1.0]])

    dist = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])

    marker_size = 13.7e-3

    if np.all(ids != None):
        print("Detected marker id: ", ids[0])
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], marker_size, mtx, dist)
        rmat = cv2.Rodrigues(rvec[0,0])

    tvec = tvec[0,0]
    rmat = rmat[0]

    print("Translation", tvec)
    print("Rotation", rmat)

    return tvec, rmat