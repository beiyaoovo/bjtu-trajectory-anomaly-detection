"""
This module provides the algorithm TRAOD and related
functions required for data preprocessing.

The algorithm TRAOD is used to detect trajectory outlier.
It was presented by Jae-Gil Lee and others in their paper
"Trajectory Outlier Detection: A Partition-and-Detect Framework."
There are two kinds of TRAOD algorithms mentioned in this paper,
one is the outlier detection algorithm TRAOD (basic),
and the other is the outlier detection algorithm TRAOD (enhanced).
Our algorithm is the basic one.

See https://www.kaistdmlab.org/_files/ugd/c5ff4f_d39dcbd2bd4f45438fa432022ce7beda.pdf
for the paper.
"""

import math
import sub_segment
import time


def LonLatToXY(lon, lat):
    """
    Match longitude and latitude data to Cartesian coordinates.

    Parameters
    ----------
    lon : int or float
    lat : int or float

    Returns
    -------
    tuple
        The converted Cartesian coordinates with (lon=0 °, lat=0 °) as the origin.
    """
    x = - lon * 40075020 * math.cos(lat * math.pi / 360) / 360
    y = lat * 40075020 / 360
    return x, y


def data_preprocessing(graph, trajectory):
    """
    Data preprocessing for TRAOD algorithm.

    Parameters
    ----------
    graph : class : `networkx.classes.multigraph.MultiGraph` object
    trajectory : list or tuple
        Elements in the list or tuple are trajectory data, and each trajectory
        is a list or tuple. The elements in the trajectory are sampling points,
        which are represented by a binary list or binary tuple. The ID in the
        graph and sampling time of the points are recorded in sequence.

    Returns
    -------
    list
    [   [[x0, y0, ...], [x1, y1, ...], ...],
        ...
    ]
        Preprocessed data. All sampling points are represented in Cartesian coordinates.
    """
    tr_xy = []
    for tr in trajectory:
        temp = []
        for item in tr:
            x, y = LonLatToXY(graph.nodes[item[0]]['x'], graph.nodes[item[0]]['y'])
            temp.append([x, y])
        tr_xy.append(temp)
    return tr_xy


def traod(graph, trajectory, d=80, p=0.95, f=0.05):
    """
    The outlier detection algorithm TRAOD (basic).

    Our algorithm TRAOD requires three parameters: D, p,
    and F. The most tricky and sensitive parameter is D. We
    suggest that users change the value of D and check the result
    repeatedly during the search for outliers. A larger value of D
    generates a smaller number of outliers, and a smaller value of
    D a larger number of outliers.
    For the parameter p, it is reasonable to select a value
    of p very close to unity since an outlier occurs relatively
    infrequently. Our experience indicates that p should be closer
    to unity as the number of trajectories |I| increases. For
    example, p = 0.95 may suffice when |I| < 10^3, but p = 0.99
    may be more appropriate when |I| ≈ 10^6.
    The parameter F represents the threshold of allowable
    noises that are not regarded as outliers. Our experience indicates
    that F should be smaller as the length of trajectories
    lenᵢ gets longer. For example, F = 0.2 may suffice when
    avg(lenᵢ) < 100, but F = 0.1 may be more appropriate when
    avg(lenᵢ) > 1000.
    (Excerpted from the paper.)

    Parameters
    ----------
    graph : class : `networkx.classes.multigraph.MultiGraph` object
    trajectory : list or tuple
        Elements in the list or tuple are trajectory data, and each trajectory
        is a list or tuple. The elements in the trajectory are sampling points,
        which are represented by a binary list or binary tuple. The ID in the
        graph and sampling time of the points are recorded in sequence.
    d, p, f : float

    Returns
    -------
    [TR0_PRE, TR1_PRE, ...] : list
        The prediction sequence of each trajectory.
        0 is normal and 1 is abnormal.
    """
    NUM = 0
    if d < 0 or not 0 <= p <= 1 or not 0 <= p <= 1:
        exit(1)
    y_score = []
    trs = data_preprocessing(graph, trajectory)
    num_trs = len(trs)
    sub_seg = []  # all sub-segments: two-dimensional list
    # [
    #   tr0: [ [tr0_seg0], [tr0_seg1], ... ]
    #   tr1: [ [tr1_seg0], [tr0_seg1], ... ]
    # ]

    # Separate sub_seg
    for i in range(len(trs)):
        tr = trs[i]
        sub_seg.append([])
        for j in range(len(tr) - 1):
            seg = sub_segment.SubSegment(tr[j], tr[j + 1], num_trs)
            sub_seg[i].append(seg)
    num_seg = 0
    for item in sub_seg:
        for i in item:
            num_seg += 1
    print(num_seg)

    pairwise_distance = []  # distance of all sub-segment pairs
    # for i in range(len(sub_seg)):
    #     time_start = time.perf_counter()
    #     for seg1 in sub_seg[i]:
    #         for j in range(len(sub_seg)):
    #             seg1.len_close_seg.append(0)
    #             seg1.count_close_seg.append(0)
    #             if i != j:  # Do not compare with sub-segments of the same trajectory.
    #                 for seg2 in sub_seg[j]:
    #                     distance = sub_segment.dist(seg1, seg2)
    #                     pairwise_distance.append(distance)
    #                     seg1.seg_distance.append(distance)
    #                     if distance < d:  # Judge whether it is an adjacent sub-segment.
    #                         seg1.count_close_seg[-1] += 1
    #                         seg1.len_close_seg[-1] += seg2.length
    #                 if seg1.len_close_seg[-1] >= seg1.length:  # Judge whether it is an adjacent trajectory.
    #                     seg1.count_close_tr += 1
    for i, tra_i in enumerate(sub_seg[: -1]):
        # time_start = time.perf_counter()
        for li in tra_i:
            for j, tra_j in enumerate(sub_seg[i + 1:]):
                index = i + j + 1
                for lj in tra_j:
                    NUM += 1
                    distance = sub_segment.dist(li, lj)
                    # print(distance)
                    li.seg_distance.append(distance)
                    lj.seg_distance.append(distance)
                    pairwise_distance.append(distance)
                    if distance < d:  # Judge whether it is an adjacent sub-segment.
                        li.len_close_seg[index] += lj.length
                        lj.len_close_seg[i] += li.length

    for index, tra in enumerate(sub_seg):
        for seg in tra:
            for i, length in enumerate(seg.len_close_seg):
                if i != index:
                    if length >= seg.length:
                        seg.count_close_tr += 1

    # Calculate the standard deviation of the sub-segment distance.
    sum_dis = 0
    for item in pairwise_distance:
        sum_dis += float(item)
    u = sum_dis / len(pairwise_distance)  # average sub-segment distance
    s = 0
    for item in pairwise_distance:
        s += (item - u) ** 2
    std = math.sqrt(s / len(pairwise_distance))  # standard deviation
    print("std={}".format(std))

    # density
    for i in range(len(sub_seg)):
        for seg in sub_seg[i]:
            for item in seg.seg_distance:
                if item < std:
                    seg.density += 1

    num_seg = 0  # total number of sub-segments
    for i in range(len(sub_seg)):
        num_seg += len(sub_seg[i])

    sum_density = 0  # sum of sub-segment density
    for i in range(len(sub_seg)):
        for item in sub_seg[i]:
            sum_density += item.density

    # Check out the outlying sub_segments.
    # True：not outlying
    # False：outlying
    for i in range(len(sub_seg)):
        for seg in sub_seg[i]:
            if seg.density == 0:
                seg.judge = True
            elif math.ceil(seg.count_close_tr * sum_density / num_seg / seg.density) \
                    <= math.ceil((1 - p) * len(sub_seg)):
                # sub-segment outlier condition：⌈|CTR(Li,D)| * adj(Li)⌉ ≤ ⌈(1- p)|I|⌉
                # adj(Li) = sum_density / num_seg / Li.density
                seg.judge = False
            else:
                seg.judge = True

    # Check out the outlying trajectories.
    for tr in sub_seg:
        len_outlying_seg = 0  # total length of outlier segments of current trajectory
        len_seg = 0  # total length of segments of current trajectory
        for seg in tr:
            len_seg += seg.length
            if not seg.judge:
                len_outlying_seg += seg.length

        if len_outlying_seg / len_seg >= f:  # trajectory outlier condition
            y_score.append(1)
        else:
            y_score.append(0)

    print(NUM)
    return y_score
