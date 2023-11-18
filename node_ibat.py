"""
This module provides the algorithm iBAT and related
functions required for data preprocessing.

The algorithm iBAT is used to detect anomalous trajectories.
It was presented by Daqing Zhang and others in their paper
"iBAT: Detecting Anomalous Taxi Trajectories from GPS Traces."

See http://www-public.tem-tsp.eu/~zhang_da/pub/ibat-ubicomp11.pdf
for the paper.
"""

import math
import random


def data_preprocessing(trajectory):
    """
    Data preprocessing for iBAT algorithm.

    Parameters
    ----------
    trajectory : list or tuple
        Elements in the list or tuple are trajectory data, and each trajectory
        is a list or tuple. The elements in the trajectory are sampling points,
        which are represented by a binary list or binary tuple. The ID in the
        graph and sampling time of the points are recorded in sequence.

    Returns
    -------
    list
    [ [point0_id, point1_id, ...], ... ]
        Preprocessed data. All sampling points are represented by node id.
    """
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


def ibat(trajectory, m=100, n=256):
    """
    An Isolation-Based Anomalous Trajectory (iBAT) detection method.

    Parameters
    ----------
    graph : class : `networkx.classes.multigraph.MultiGraph` object
    trajectory : list or tuple
        Elements in the list or tuple are trajectory data, and each trajectory
        is a list or tuple. The elements in the trajectory are sampling points,
        which are represented by a binary list or binary tuple. The ID in the
        graph and sampling time of the points are recorded in sequence.
    lon_min, lat_min, lon_max, lat_max : float
        The bounds of the graph.
    m : int
        Number of experiments.
    n : int
        The size of the subsample.

    Returns
    -------
    [TR0_PRE, TR1_PRE, ...] : list
        The prediction sequence of each trajectory.
        0 is normal and 1 is abnormal.
    """
    trs = data_preprocessing(trajectory)
    y_score = []
    for tr in trs:
        num = [0 for item in range(m)]  # Record the separation times of each experiment.
        for j in range(m):
            # All nodes in the trajectory to be measured are out of order.
            # Simulate randomly taking the element p each time.
            p_cells = random.sample(tr, len(tr))
            # Take out the sub-sample of size n.
            tr_sample = random.sample(trs, n)
            p = -1
            while len(tr_sample) > 1:
                p += 1
                if p >= len(tr):
                    break
                num[j] += 1
                # Check all trajectories containing the cell p in the subsample and delete them.
                for k in range(len(tr_sample) - 1, -1, -1):
                    if p_cells[p] not in tr_sample[k]:
                        tr_sample.pop(k)
        # Calculate the anomaly score.
        sum_num = 0
        for item in num:
            sum_num += item
        e_num_t = sum_num / m
        c_n = 2 * (math.log(n - 1) + 0.57721566) - 2 * (n - 1) / n
        s_t_n = pow(2, -e_num_t / c_n)
        if e_num_t > c_n and s_t_n < 0.5:
            y_score.append(0)
        else:
            y_score.append(1)
    return y_score
