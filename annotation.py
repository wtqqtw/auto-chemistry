import numpy as np
from matplotlib import pyplot as plt
import os
import PIL

dir_path = 'dataset_perspective_transformed/'
files = os.listdir(dir_path)

for file in files:
    try:
        img = PIL.Image.open(dir_path + file)
    except PIL.UnidentifiedImageError:
        continue
    plt.imshow(img), plt.xticks([]), plt.yticks([]), plt.title("visually count # points, then close the window")
    plt.show()
    n = int(input(f"{file}, # points: "))
    plt.imshow(img), plt.title(file)
    coord = plt.ginput(n, timeout=100, show_clicks=True)
    arr = np.array(coord).astype(int)
    np.save(f'ground_truth_coordinates/{file.removesuffix(".png")}.npy', arr)
    plt.close()
