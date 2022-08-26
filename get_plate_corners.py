import numpy as np
from matplotlib import pyplot as plt
import os
import PIL

dir_path = 'dataset_cropped/'
files = os.listdir(dir_path)

# mark 4-points on each cropped image, uncomment this part if you want to re-mark
for i, file in enumerate(files):
    try:
        img = PIL.Image.open(dir_path + file)
    except PIL.UnidentifiedImageError:
        continue
    plt.imshow(img), plt.title(file)
    coord = plt.ginput(4, timeout=100, show_clicks=True)
    arr = np.array(coord).astype(int)
    np.save(f'perspective_trans_points/{file.removesuffix(".png")}.npy', arr)
    plt.close()
