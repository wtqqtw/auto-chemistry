import os
import math
import numpy as np
from numpy import linalg as la
import utils
from detector import find_spots

th = 0.02


def exist_dupli(assign: list):
    elems = set()
    for elem in [x[0] for x in assign if x is not None]:
        if elem in elems:
            return True
        else:
            elems.add(elem)
    return False


def not_complete(assign: list):
    if len(assign) < 1:
        return True
    if set(assign) == {None}:
        return False
    if exist_dupli(assign):
        return True
    else:
        return False


def evaluate(T, P):
    D = {}
    for t in range(len(T)):
        for p in range(len(P)):
            d = la.norm(T[t] - P[p])
            if t in D:
                D[t].append((p, d))
            else:
                D[t] = [(p, d)]
    for t in D:
        D[t].sort(key=lambda x: x[-1], reverse=True)

    assign = []
    while not_complete(assign):
        if len(assign) < 1:
            for l in D.values():
                if l[-1][-1] < th * math.sqrt(w ** 2 + h ** 2):
                    assign.append(l.pop(-1))
                else:
                    assign.append(None)
        else:
            unique = set([x[0] for x in assign if x is not None])
            for u in unique:
                dupli = []
                for i, x in enumerate(assign):
                    if x is not None and x[0] == u:
                        dupli.append((assign[i], i))
                dupli.sort(key=lambda x: x[0][-1], reverse=True)
                for x, i in dupli[:-1]:
                    if len(D[i]) > 0:
                        assign[i] = D[i].pop(-1)
                    else:
                        assign[i] = None

    assert not exist_dupli(assign)

    recall = 1 - assign.count(None) / len(assign)
    precision = (len(assign) - assign.count(None)) / len(P)

    return recall, precision


if __name__ == '__main__':
    path = 'dataset_perspective_transformed/'
    files = os.listdir(path)

    # those are images which we can not detect margin lines from
    neg = ['014.png', '017.png', '026.png', '028.png', '031.png', '032.png', '036.png']
    files = set(files) - set(neg)

    recall = []
    precision = []
    for f in files:
        im, _ = utils.imread(path + f)
        h, w, _ = im.shape
        T = np.load('eval_data/' + f.split('.')[0] + '.npy')
        P, rfs = find_spots(im)
        # print(T.shape, P.shape)
        r, p = evaluate(T, P)
        recall.append(r)
        precision.append(p)
        print(f'{f}, Recall: {r}, Precision: {p}')

    mean_recall = np.array(recall).mean()
    mean_precision = np.array(precision).mean()

    print('\n\n')
    print(f'mean recall: {mean_recall}, mean precision: {mean_precision}')

# Result
# mean recall: 0.8037513412513413, mean precision: 0.8423708866890686
