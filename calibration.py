import cv2 as cv
import numpy as np
import utils
from scipy.spatial import distance as dist
import os
from matplotlib import pyplot as plt

def corner_st(im, n_grids=8):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    gray = utils.denoise(gray)
    h, w = gray.shape
    mask = np.ones(gray.shape, np.uint8)
    h_lim = [int(h / n_grids), int(h * (1 - n_grids) / n_grids)]
    w_lim = [int(w / n_grids), int(w * (1 - n_grids) / n_grids)]
    mask[h_lim[0]:h_lim[1], w_lim[0]:w_lim[1]] = 0

    corners = cv.goodFeaturesToTrack(image=gray, maxCorners=50, qualityLevel=0.1, minDistance=w / 2, mask=mask)
    corners = np.int0(corners)

    rect = cv.minAreaRect(corners)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    return box


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    calibrated = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    return calibrated


def calibrate(im):
    pts = corner_st(im)
    return transform(im, pts)

if __name__ == '__main__':
    # mask = np.ones((8,8), np.uint8)
    # h_lim = [int(1), int(7)]
    # w_lim = [int(1), int(7)]
    # mask[h_lim[0]:h_lim[1], w_lim[0]:w_lim[1]] = 0
    # print(mask)


    path = 'dataset_cropped/'
    files = os.listdir(path)
    n_grids = 4
    for f in files:
        if f == '.DS_Store':
            continue
        im, gray = utils.imread(path + f)

        gray = utils.denoise(gray)
        h, w = gray.shape
        mask = np.ones(gray.shape, np.uint8)
        h_lim = [int(h / n_grids), int(h * (n_grids - 1) / n_grids)]
        w_lim = [int(w / n_grids), int(w * (n_grids - 1) / n_grids)]
        mask[h_lim[0]:h_lim[1], w_lim[0]:w_lim[1]] = 0
        # print(mask[int(h/2),int(w/2)])

        corners = cv.goodFeaturesToTrack(image=gray, maxCorners=50, qualityLevel=0.05, minDistance=w / 5, mask=mask)
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            cv.circle(im, (x, y), 3, 255, -1)

        rect = cv.minAreaRect(corners)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(im, [box], 0, (255, 0, 0), 2)
        plt.imshow(im), plt.show()