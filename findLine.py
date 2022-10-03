import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import os
from sklearn.cluster import DBSCAN
import utils


def l_eq(line):
    """given 2 points, calculate line equation y=mx+c"""
    points = get_points([line])[0]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
    x = w / 2
    y = m * x + c
    return m, c, y


def get_points(lines):
    """calculate startpoint and endpoint by rho and theta"""
    P = None
    for line in lines:
        rho, theta, vote = line
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        if P is None:
            P = [(pt1, pt2)]
        else:
            P.append((pt1, pt2))
    return P


def k_rep(indices, lines):
    """select representative of cluster k"""
    if len(lines) > 0:
        k_lines = [lines[i] for i in indices]
        if len(k_lines) > 0:
            maxVote = max([l[-1] for l in k_lines])
            for l in k_lines:
                if l[-1] == maxVote:
                    return l
    return None


def optimize_line(lines):
    """average some crossed lines and closed lines"""
    smoothed = []
    optimized = []
    for i, l1 in enumerate(lines):
        if i not in smoothed:
            similar = []
            for j, l2 in enumerate(lines):
                if i == j or (math.fabs(l_eq(l2)[-1] - l_eq(l1)[-1]) < 0.02 * h):
                    smoothed.append(j)
                    similar.append(l2)

            line_arr = np.array([l for l in similar])
            vote = line_arr[:, -1]
            total = np.sum(vote)
            weightByVote = vote / total
            weightedMean = list(weightByVote @ line_arr)
            optimized.append(weightedMean)
    return optimized


def find_objective_lines(lines):
    """find upper and lower bound"""
    Y = [[l_eq(l)[-1], l] for l in lines]
    Y.sort(key=lambda x: x[0])
    gaps = [Y[i + 1][0] - Y[i][0] for i in range(len(Y) - 1)]
    maxGap = max(gaps)
    idx = gaps.index(maxGap)
    return [Y[idx][1], Y[idx + 1][1]]


def hough_line(src):
    binary = utils.binarize(utils.denoise(src))
    global h, w
    h, w = binary.shape
    houghLines = cv.HoughLinesWithAccumulator(binary, 1, np.pi / 180, int(w * 0.4))
    lines = []
    if houghLines is not None:
        for i in range(len(houghLines)):
            theta = houghLines[i][0][1]
            if 5 / 12 <= theta / np.pi <= 7 / 12:  # >= 75 and <= 105 degree
                lines.extend(houghLines[i])

    # dbscan clustering by (rho,theta)

    # rho = math.sqrt(h ** 2 + w ** 2)
    # theta = np.pi
    X = np.array([[l[0], (l[1] / np.pi) * 180] for l in lines])
    db = DBSCAN(eps=math.sqrt(10 ** 2), min_samples=2).fit(X)
    core_indices = set(db.core_sample_indices_)
    labels_ = db.labels_
    K = set(labels_) - {-1}
    K_reps = []
    for k in K:
        k_indices = [i for i, x in enumerate(labels_) if x == k]
        kRep = k_rep(k_indices, lines)
        if kRep is not None:
            K_reps.append(kRep)

    outliers = [lines[i] for i, x in enumerate(labels_) if x == -1]
    optimized_lines = optimize_line(outliers + K_reps)
    objective_lines = find_objective_lines(optimized_lines)
    # points = get_points(objective_lines)
    return objective_lines


if __name__ == '__main__':
    dirPath = 'dataset_perspective_transformed/'
    f = os.listdir(dirPath)
    # f = ['dataset_perspective_transformed/002.png']
    for fileName in f:

        im, gray = utils.imread(dirPath + fileName)

        objective_lines = hough_line(gray)
        points = get_points(objective_lines)
        for pt1, pt2 in points:
            cv.line(im, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
        plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
        plt.title(f'{fileName},annotate lines')
        plt.show()
        # cv.imwrite('lines/' + fileName, im)
