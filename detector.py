import os

import cv2 as cv
import numpy as np
import numpy.linalg as la
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import utils
import keyPoints as kp
from retention_factor import rf

bar_confident_cnt = 1e-2
band_width = 4
lower_ratio = 1e-2
upper_ratio = 0.5
quantile = 0.9
theta = 5
denoise_kernel = 5
outlier_min_samples = 10


def find_spots(im: np.ndarray):
    global gray, h, w
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    gray = utils.denoise(gray, denoise_kernel)
    msk = utils.mask(gray)
    contours, hierarchy = cv.findContours(image=utils.binarize(gray), mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_SIMPLE)

    contours = optimise_contour(contours, msk)
    KP = kp.orb(gray)
    pt = np.int0(np.array([p.pt for p in KP]))

    assign = assign_kp(contours, pt)
    outliers = pt[[pt_id for pt_id in range(len(pt)) if pt_id not in assign]]

    contours = confident_cnt(contours, assign)
    P = optimise_circle(contours)
    Q = kp_cluster(outliers)

    if Q is not None:
        P = np.vstack((P, np.array(Q)))

    # dst = im.copy()
    # for o in P:
    #     cv.drawMarker(img=dst, position=(int(o[0]), int(o[1])), color=(0, 0, 255), markerType=cv.MARKER_CROSS)
    # plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB)), plt.show()
    # cv.imwrite(f'spots/{f}', dst)

    rfs = rf(P,w)
    return P, rfs


def kp_cluster(outliers: np.ndarray):
    P = []
    eps = la.norm(np.array([lower_ratio * w, lower_ratio * h]))
    clustering = DBSCAN(eps=eps, min_samples=outlier_min_samples).fit(outliers)
    labels = np.array(clustering.labels_)
    K = set(labels) - {-1}
    indices = [np.argwhere(labels == k) for k in K]
    for i, k in enumerate(indices):
        p = outliers[k].squeeze(axis=1).mean(axis=0)
        P.append(p)
    if len(P) > 1:
        return np.array(P)
    return None


def optimise_circle(C: list):
    pms_circle = []
    pms_cnt = []
    for cnt in C:
        circle = utils.fit_circle(cnt)
        cx, cy, r = circle
        if 2 * lower_ratio * min(w, h) <= r <= upper_ratio * min(w, h):
            pms_circle.append(circle)
            pms_cnt.append(cnt)

    pms_circle = np.array(pms_circle)

    eps = la.norm(np.array([lower_ratio * w, lower_ratio * h]))
    clustering = DBSCAN(eps=eps, min_samples=1).fit(pms_circle[..., :-1])
    labels = np.array(clustering.labels_)
    K = set(labels)

    P = np.zeros((len(K), 2))
    indices = [np.argwhere(labels == k) for k in K]
    for i, k in enumerate(indices):
        circles = pms_circle[k].squeeze(axis=1)
        if len(circles) > 1:
            P[i] = np.mean(circles[..., :-1], axis=0)
        else:
            cx, cy, r = circles[0]
            circle_centre = np.array([cx, cy])
            cnt_centre = np.array(utils.centre_cnt(pms_cnt[k.squeeze()]))
            if cnt_centre is not None:
                l2 = la.norm(circle_centre - cnt_centre)
                if l2 / r > 0.5:
                    P[i] = cnt_centre
                else:
                    P[i] = (cnt_centre + circle_centre) / 2
    return P


def confident_cnt(contour: list, assign: dict):
    C = []
    for i, cnt in enumerate(contour):
        count = np.sum([i in l for l in assign.values()])
        ratio = count / len(cnt)
        if ratio >= bar_confident_cnt:
            C.append(cnt)
    return C


def boundary_check(cnt: np.ndarray, msk: np.ndarray):
    h_bound = False
    v_bound = False
    bw = band_width * lower_ratio
    x, y = cnt.T
    p = np.sum(msk[[y], [x]]) / len(cnt)
    if p < 0.5:
        h_bound = True
    if np.mean(x) < bw * w or np.mean(x) > (1 - bw) * w:
        if np.std(x) < lower_ratio * w:
            v_bound = True
    return h_bound + v_bound


def optimise_rect(C: list):
    rect = [cv.minAreaRect(cnt) for cnt in C]
    shape = np.array([r[1] for r in rect])
    ar = np.max(shape, axis=1) / np.min(shape, axis=1)
    area = np.prod(shape, axis=1)
    ar = (ar - np.min(ar)) / np.ptp(ar)
    area = 1 - (area - np.min(area)) / np.ptp(area)

    l = area + theta * ar
    bar = np.quantile(l, q=quantile)

    indices = np.argwhere(l < bar).squeeze().astype(int)
    C = [C[i] for i in indices]
    rect = [rect[i] for i in indices]
    return C, rect


def optimise_contour(contours: tuple, msk: np.ndarray):
    C = []
    for i, cnt in enumerate(contours):
        cnt = cnt.squeeze(axis=1)
        if not boundary_check(cnt, msk):
            C.append(cnt)

    # exclude very small contours: noise points
    cnt_area = np.array([cv.contourArea(cnt) for cnt in C])
    min_area = lower_ratio ** 2 * (w * h)
    indices = np.argwhere((cnt_area > min_area)).squeeze()
    C = [C[i] for i in indices]

    # exclude contours that have high aspect ratio or very small rect:
    C, rect = optimise_rect(C)

    return C


def assign_kp(C: list, KP: np.ndarray):
    assign = {}
    for pt_id, kp in enumerate(KP):
        for i, cnt in enumerate(C):
            MAE = np.abs(cnt - kp).sum(axis=1)
            if np.min(MAE) < lower_ratio * (w + h):
                if pt_id not in assign:
                    assign[pt_id] = [i]
                else:
                    assign[pt_id].append(i)
    return assign


if __name__ == '__main__':
    path = 'dataset_perspective_transformed/'
    files = os.listdir(path)
    for f in files:
        im, _ = utils.imread(path + f)
        P, rfs = find_spots(im)
        print(P), print(rfs)
