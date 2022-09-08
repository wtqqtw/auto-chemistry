import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import os
from sklearn.cluster import DBSCAN

# epsilon = 0.01

dirPath = 'dataset_perspective_transformed/'
files = os.listdir(dirPath)
# files = ['002.png']
for fileName in files:
    img = cv.imread(dirPath + fileName)
    h, w, _ = img.shape
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # # cv.fastNlMeansDenoisingColored: perfectly removed noise, but also removed thin(small) lines
    # dst = cv.fastNlMeansDenoisingColored(img,None,h=3,hColor=3,templateWindowSize=7,searchWindowSize=21)
    # plt.subplot(121),plt.imshow(img)
    # plt.subplot(122),plt.imshow(dst)
    # plt.show()
    #
    # # grayscale cv.fastNlMeansDenoising with same issue
    # img = cv.imread('dataset_perspective_transformed/002.png',0)
    # dst = cv.fastNlMeansDenoising(gray,None,h=3,templateWindowSize=7,searchWindowSize=21)
    # plt.subplot(121),plt.imshow(img,cmap='gray')
    # plt.subplot(122),plt.imshow(dst,cmap='gray')
    # plt.show()
    #
    # # bitateral blurring: removed some noise, while keeping the edges clear
    # blur = cv.bilateralFilter(img,d=5,sigmaColor=100,sigmaSpace=100)
    # plt.subplot(121),plt.imshow(img)
    # plt.subplot(122),plt.imshow(blur)
    # plt.show()

    # step 1. add contrast
    # blur = cv.medianBlur(gray,3)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization), better than standard way
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)



    # step 2. de-noising
    # morphology opening: removed (salt) noise, while keep edges clear
    kernel = np.ones((7, 7), np.uint8)
    opening = cv.morphologyEx(cl, op=cv.MORPH_OPEN, kernel=kernel)


    # step 3. convert to binary image

    # adaptive thresholding, better than Canny Edge
    th = cv.adaptiveThreshold(opening, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # remove (pepper) noise
    dst = cv.medianBlur(th, 3)
    dst_invert = cv.bitwise_not(dst)

    # display results
    plt.subplot(121), plt.imshow(gray, cmap='gray'), plt.title('original image')
    # plt.subplot(122), plt.imshow(opening, cmap='gray'), plt.title('2.denoise')
    # plt.show()
    # plt.subplot(121), plt.imshow(cl, cmap='gray'), plt.title('3.add contrast')
    plt.subplot(122), plt.imshow(dst_invert, cmap='gray'), plt.title('binary')
    plt.show()


    # step4. annotate bottom line and upper line

    # Hough Transform find the potential lines (have # pixels > w/3)
    houghLines = cv.HoughLinesWithAccumulator(dst_invert, 1, np.pi / 180, int(w * 0.4))
    lines = []
    if houghLines is not None:
        for i in range(len(houghLines)):
            theta = houghLines[i][0][1]
            if 5 / 12 <= theta / np.pi <= 7 / 12:  # >= 75 and <= 105 degree
                lines.extend(houghLines[i])


    def l_eq(line):
        """given 2 points, calculate line equation y=mx+c"""
        points = get_points([line])[0]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
        x = w / 2
        y = m * x + c
        return m,c,y


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
                P = [(pt1,pt2)]
            else:
                P.append((pt1,pt2))
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


    # dbscan clustering by (rho,theta)
    rho = math.sqrt(h ** 2 + w ** 2)
    theta = np.pi
    X = np.array([[l[0], (l[1] / np.pi) * 180] for l in lines])
    db = DBSCAN(eps=math.sqrt(10**2), min_samples=2).fit(X)
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

    def optimize_line(lines):
        """average some crossed lines and closed lines"""
        smoothed = []
        optimized = []
        for i, l1 in enumerate(lines):
            if i not in smoothed:
                similar = []
                for j, l2 in enumerate(lines):
                    if i==j or (math.fabs(l_eq(l2)[-1]-l_eq(l1)[-1])<0.02*h):
                        smoothed.append(j)
                        similar.append(l2)

                line_arr = np.array([l for l in similar])
                vote = line_arr[:,-1]
                total = np.sum(vote)
                weightByVote = vote/total
                weightedMean = list(weightByVote@line_arr)
                optimized.append(weightedMean)
        return optimized

    def find_objective_lines(lines):
        """find upper and lower bound"""
        Y = [[l_eq(l)[-1],l] for l in lines]
        Y.sort(key=lambda x: x[0])
        gaps = [Y[i+1][0]-Y[i][0] for i in range(len(Y)-1)]
        maxGap = max(gaps)
        idx = gaps.index(maxGap)
        return [Y[idx][1],Y[idx+1][1]]




    optimized_lines = optimize_line(outliers+K_reps)
    objective_lines = find_objective_lines(optimized_lines)
    points = get_points(objective_lines)


    for pt1, pt2 in points:
        cv.line(img, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(f'{fileName},annotate lines')
    plt.show()
    cv.imwrite('lines/' + fileName, img)

