import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PT import transform

dir_path = 'dataset_cropped/'
files = os.listdir(dir_path)

# uncomment this part if you want to re-generate perspective-transformed images
for file in files:
    img = cv2.imread(dir_path + file)
    arr_path = 'perspective_trans_points/' + file.removesuffix('png') + 'npy'
    try:
        arr = np.load(arr_path)
    except FileNotFoundError:
        continue
    warped = transform(img, arr)
    plt.imshow(warped)
    plt.xticks([]), plt.yticks([]), plt.title(file)
    plt.show()
    cv2.imwrite('dataset_perspective_transformed/' + file, warped)
