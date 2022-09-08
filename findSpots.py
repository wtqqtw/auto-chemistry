import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import os
from sklearn.cluster import DBSCAN

img = cv.imread('dataset_perspective_transformed/002.png')
h, w, _ = img.shape
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def pre_process(src, blur=3):
    blur = cv.medianBlur(src, blur)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(blur)

    kernel = np.ones((7, 7), np.uint8)
    opening = cv.morphologyEx(cl, op=cv.MORPH_OPEN, kernel=kernel)

    return opening


def gradient(src, ksize):
    laplacian = cv.Laplacian(src, ddepth=-1, ksize=ksize, scale=.1)

    sobelx = cv.Sobel(src, ddepth=-1, dx=2, dy=0, ksize=ksize, scale=.1)
    sobely = cv.Sobel(src, ddepth=-1, dx=0, dy=2, ksize=ksize, scale=.1)
    sobel = cv.addWeighted(sobelx, .5, sobely, .5, 0)

    scharrx = cv.Scharr(src, ddepth=-1, dx=1, dy=0, scale=2)
    scharry = cv.Scharr(src, ddepth=-1, dx=0, dy=1, scale=2)
    scharr = cv.addWeighted(scharrx, .5, scharry, .5, 0)

    canny = cv.Canny(src,threshold1=10,threshold2=100,apertureSize=3,L2gradient=False)

    plt.subplot(141), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
    plt.subplot(142), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
    plt.subplot(143), plt.imshow(scharr, cmap='gray'), plt.title('Scharr')
    plt.subplot(144), plt.imshow(canny, cmap='gray'), plt.title('Canny')
    plt.show()

def binarize(src,blur=3):
    th = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    dst = cv.medianBlur(th, blur)
    plt.imshow(cv.bitwise_not(dst),cmap='gray'), plt.title('Threshold'),plt.show()


gradient(pre_process(gray), 7)
binarize(pre_process(gray,blur=5))
