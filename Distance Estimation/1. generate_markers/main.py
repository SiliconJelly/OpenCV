import cv2 as cv
from cv2 import aruco

# dictionary to specify type of the marker
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

# MARKER_ID = 0
MARKER_SIZE = 400  # pixels

# generating unique IDs using for loop
for id in range(20):  # genereting 20 markers
    # using funtion to draw a marker
    marker_image = aruco.generateImageMarker(marker_dict, id, MARKER_SIZE)
    cv.imshow("img", marker_image)
    cv.imwrite(f"markers/marker_{id}.png", marker_image)
    # cv.waitKey(0)
    # break
