import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

img_path = 'dataset_perspective_transformed/'
files = os.listdir(img_path)

for f in files:
    img = cv2.imread(img_path+f)
    coords_arr = 'ground_truth_coordinates/' + f.split('.')[0] + '.npy'
    coords = np.load(coords_arr)
    # print(coords.shape)
    for coord in coords:
        cv2.drawMarker(img,(coord[0], coord[1]),color=(0,0,255), markerType=cv2.MARKER_SQUARE, markerSize=20)

    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([]), plt.title(f)
    plt.show()
    cv2.imwrite('marked_images/'+f,img)