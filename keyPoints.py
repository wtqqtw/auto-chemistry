import cv2 as cv
import utils


def orb(gray):
    orb = cv.ORB_create(nfeatures=2000, scaleFactor=1.1, nlevels=8, edgeThreshold=11, firstLevel=0, patchSize=11,
                        fastThreshold=5)
    msk = utils.mask(gray)
    KP, des = orb.detectAndCompute(gray, msk)
    return KP


def sift(gray):
    sift = cv.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)
    msk = utils.mask(gray)
    KP, des = sift.detectAndCompute(gray, msk)
    return KP


def fast(gray):
    fast = cv.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16)
    msk = utils.mask(gray)
    KP = fast.detect(gray, msk)
    return KP
