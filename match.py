import math
import numpy as np

def distance(ground,detect,threshold):
    dist = []
    for p in ground:
        for q in detect:
            d = math.dist(p,q)
            dist.append(math.dist(p,q))
    matrix = np.asarray(dist).reshape(ground.shape[0],detect.shape[0])
    return matrix

def match(dist,threshold):
    pair = []
    count = 0
    for i in dist.shape[0]:
        d_index = dist[i].argmin()
        d_value = dist[i].min()
        if d_value > threshold:
            pair[i] = -1
        else:
            pair[i] = d_index
            # if d_index not in pair :
            #     pair[i] = d_index
            # else:
            #     point = pair.index(d_index)
            #     if dist[point][d_index]>d_value: # the new point is closer compare with the previous one
            #
        result = np.asarray(pair)
        while(count<len(pair)):
            for i in len(pair):
                if pair[i]==-1 or pair.count(pair[i])==1:
                    count+=1
                else:
                    equal_dist_points = np.argwhere(result == pair[i])
                    closest_point = np.asarray([dist[i][x] for x in equal_dist_points[:,0]]).argmin()
                    for p in equal_dist_points[:,0]:
                        if p != closest_point:
                            dist[i][p]=math.inf


def recall(match_num,total_num):
    return match_num/total_num






