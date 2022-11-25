import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit
import os
import utils
import warnings

warnings.filterwarnings('ignore')

beta = 2
std_th = 0.06
perp_th = 0.04


def total_ls(M: np.ndarray):
    m, _ = M.shape
    A = M - np.mean(M, axis=0)

    w, v = np.linalg.eig(A.T @ A)
    p = v[np.argmin(w)]

    a, b = p
    c = -p @ np.mean(M, axis=0)
    theta = np.array([a, b, c])

    _M = np.hstack((M, np.ones((m, 1))))
    D = np.abs(_M @ theta)

    return (a, b, c), D


def cluster(M: np.ndarray):
    l = []
    m, n = M.shape
    x, y = M.T
    sigma = np.std(x)

    if sigma <= std_th * w:
        l.extend([M])
    else:
        eps = sigma / beta
        db = DBSCAN(eps=eps, min_samples=1).fit(x.reshape((m, 1)))
        labels_ = db.labels_
        K = set(labels_)
        if K != {0}:
            for k in K:
                M_k = M[np.argwhere(labels_ == k).squeeze(axis=1)]
                l.extend(cluster(M_k))
        else:
            l.extend([M])

    return l


def ls(M: np.ndarray):
    def func(y, a, b, c):
        return (-b * y - c) / a

    x, y = M.T
    popt = curve_fit(func, y, x)[0]
    return popt


def perp(C: np.ndarray, popt: np.ndarray):
    m, n = C.shape
    C_1 = np.hstack((C, np.ones((m, 1))))
    d = np.abs(C_1 @ np.array(popt)) / np.linalg.norm(np.array(popt[:-1]))
    return d


def opt(clusterList: list):
    indices = []
    cl = []
    if len(clusterList) == 1:
        return clusterList
    for i in range(len(clusterList)):
        if i in indices:
            continue

        C = clusterList[i]
        if i == len(clusterList) - 1:
            cl.append(C)
        else:
            C_r = np.vstack((C, clusterList[i + 1]))
            if len(C_r) < 3:
                cl.append(C)
            else:
                d = perp(C_r, ls(C_r))
                if np.mean(d) < perp_th * w:
                    indices.extend([i, i + 1])
                    cl.append(C_r)
                else:
                    cl.append(C)

    for i, C in enumerate(cl):
        x,_ = np.mean(C,axis=0)
        # lineCoefficients = np.array([u,l])
        y = L@np.array([x,1])
        P = np.stack((np.array([x,x]),y))
        cl[i] = np.vstack((C,P.T))

    return cl


def rf_calc(C: np.ndarray):
    _, y = C.T
    y = np.abs(y - np.max(y))
    y.sort()
    y = y / np.max(y)

    return np.round(y[(0 < y) & (y < 1)], decimals=3)


def rf(M: np.ndarray, width, line_coefficient):
    global w; global L
    w = width; L = line_coefficient
    rfs = []
    cl = cluster(M)
    cl.sort(key=lambda x: np.mean(x, axis=0)[0])
    cl = opt(cl)
    for c in cl:
        rfs.append(rf_calc(c))
    return rfs


if __name__ == '__main__':
    imDir = 'dataset_perspective_transformed/'
    arrDir = 'ground_truth_coordinates/'
    files = os.listdir(imDir)

    import findLine as fl

    for f in files:
        im, gray = utils.imread(imDir + f)
        h, w, _ = im.shape
        u, l = fl.hough_line(gray)
        um, uc, _ = fl.l_eq(u)
        lm, lc, _ = fl.l_eq(l)
        lineCoefficients = np.array([(um,uc), (lm,lc)])
        M = np.load(arrDir + f.split('.')[0] + '.npy')
        rfs = rf(M, w,lineCoefficients)
        print(rfs)
