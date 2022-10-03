import cv2 as cv
import numpy as np
from scipy import optimize
import findLine as fl



def imread(filename):
    im = cv.imread(filename)
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    return im, gray


def denoise(src, blur=3):
    blur = cv.medianBlur(src, blur)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(blur)
    kernel = np.ones((7, 7), np.uint8)
    opening = cv.morphologyEx(cl, op=cv.MORPH_OPEN, kernel=kernel)
    return opening


def binarize(src, blur=3):
    th = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    dst = cv.medianBlur(th, blur)
    dst_invert = cv.bitwise_not(dst)
    return dst_invert


def mask(gray):
    h, w = gray.shape
    msk = np.zeros(gray.shape, dtype=np.uint8)
    u, l = fl.hough_line(gray)
    um, uc, _ = fl.l_eq(u)
    lm, lc, _ = fl.l_eq(l)
    x = np.arange(w)
    y_u = np.int0(um * x + uc)
    y_l = np.int0(lm * x + lc)
    for i, col in enumerate(msk.T):
        u, l = y_u[i], y_l[i]
        if u > 0 and l < h:
            col[u:l] = 1
    return msk


def centre_cnt(cnt: np.ndarray):
    M = cv.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy
    return None


def fit_circle(cnt: np.ndarray):
    def dist(c):
        cx, cy = c
        return np.sqrt(np.power(x - cx, 2) + np.power(y - cy, 2))

    def f(c):
        d = dist(c)
        r = np.mean(d)
        return d - r

    x, y = cnt.T
    c = np.mean(cnt, axis=0)

    output = optimize.leastsq(f, c)
    cx, cy = output[0]
    r = np.mean(dist((cx, cy)))
    circle = cx, cy, r

    return circle



