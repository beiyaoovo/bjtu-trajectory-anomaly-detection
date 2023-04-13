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
import t_partition


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


def data_preprocessing(trajectory):
    tr_id = []
    for tr in trajectory:
        if tr is None:
            continue
        temp = []
        try:
            for item in tr:
                temp.append(item[0])
        except:
            pass
        tr_id.append(temp)
    return tr_id


def enhanced_traod(trajectory, d=80, p=0.95, f=0.05):
    """
    The outlier detection algorithm TRAOD (enhanced).

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
    if d < 0 or not 0 <= p <= 1 or not 0 <= f <= 1:
        raise ValueError("Parameter error.")
    y_score = []
    trs = data_preprocessing(trajectory)
    num_trs = len(trs)
    coarse = []  # coarse partition
    fine = []  # fine partition
    num_c = 0
    # coarse partition
    for index, tra in enumerate(trs):
        cp_id = t_partition.approximate_t_partitioning(tra)
        print(cp_id)
        coarse.append([t_partition.CoarseTPartition(tra[cp_id[i]: cp_id[i + 1] + 1], num_trs, True) for i in
                       range(len(cp_id) - 1)])
        num_c += len(coarse[index])
        # print(number)

    print(num_c)
    pairwise_distance = []  # distance of all fine-t-partition pairs
    for i, tra_i in enumerate(coarse[: -1]):
        # time_start = time.perf_counter()
        for li_c in tra_i:
            for j, tra_j in enumerate(coarse[i + 1:]):
                index = i + j + 1
                for lj_c in tra_j:
                    if t_partition.lb_dist(li_c, lj_c) > d:
                        continue
                    elif t_partition.ub_dist(li_c, lj_c) <= d:
                        for li_f in li_c.subsegment:
                            if not li_f.enter:
                                fine.append(li_f)
                                li_f.enter = True
                            for lj_f in lj_c.subsegment:
                                if not lj_f.enter:
                                    fine.append(lj_f)
                                    lj_f.enter = True
                                li_f.len_close_seg[index] += lj_f.length
                                lj_f.len_close_seg[i] += li_f.length
                                distance = t_partition.dist(li_f, lj_f)
                                li_f.seg_distance.append(distance)
                                lj_f.seg_distance.append(distance)
                                pairwise_distance.append(distance)
                    else:
                        for li_f in li_c.subsegment:
                            if not li_f.enter:
                                fine.append(li_f)
                                li_f.enter = True
                            for lj_f in lj_c.subsegment:
                                if not lj_f.enter:
                                    fine.append(lj_f)
                                    lj_f.enter = True
                                distance = t_partition.dist(li_f, lj_f)
                                li_f.seg_distance.append(distance)
                                lj_f.seg_distance.append(distance)
                                pairwise_distance.append(distance)
                                if distance < d:  # Judge whether it is an adjacent sub-segment.
                                    li_f.len_close_seg[index] += lj_f.length
                                    lj_f.len_close_seg[i] += li_f.length

    for index, tra in enumerate(coarse):
        for c_t in tra:
            for f_t in c_t.subsegment:
                if f_t.enter:
                    for i, length in enumerate(f_t.len_close_seg):
                        if i != index:
                            if length >= f_t.length:
                                f_t.count_close_tr += 1

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
    for seg in fine:
        for item in seg.seg_distance:
            if item < std:
                seg.density += 1

    num_seg = len(fine)  # total number of sub-segments
    sum_density = 0  # sum of sub-segment density
    for seg in fine:
        sum_density += seg.density

    # Check out the outlying sub_segments.
    # True：not outlying
    # False：outlying
    for seg in fine:
        # print(seg.density)
        if seg.density == 0:
            seg.judge = True
        elif math.ceil(seg.count_close_tr * sum_density / num_seg / seg.density) \
                <= math.ceil((1 - p) * len(trs)):
            # sub-segment outlier condition：⌈|CTR(Li,D)| * adj(Li)⌉ ≤ ⌈(1- p)|I|⌉
            # adj(Li) = sum_density / num_seg / Li.density
            seg.judge = False
        else:
            seg.judge = True

    # Check out the outlying trajectories.
    for tra in coarse:
        len_outlying_seg = 0  # total length of outlier segments of current trajectory
        len_seg = 0  # total length of segments of current trajectory
        for seg in tra:
            for item in seg.subsegment:
                len_seg += item.length
                if not item.judge:
                    len_outlying_seg += item.length
        if len_outlying_seg / len_seg >= f:  # trajectory outlier condition
            y_score.append(1)
        else:
            y_score.append(0)

    return y_score
