"""
This module provides the algorithm T-DBSCAN and related
functions required for data preprocessing.


The algorithm T-DBSCAN is used to detect trajectory outlier.
It was presented by Zhongjian Lv and others in their paper
"Outlier Trajectory Detection: A Trajectory Analytics Based Approach,"
in DASFAA. Springer, 2017, pp 231–246.
"""


def data_preprocessing(trajectory):
    """
    Data preprocessing for iT-DBSCAN algorithm.

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
        temp = []
        for item in tr:
            temp.append(item[0])
        tr_id.append(temp)
    return tr_id


def misDistance(tra1, tra2):
    """
    Calculate the best match distance between two trajectory.

    Parameters
    ----------
    tra1 : list or tuple
    tra2 : list or tuple
        Elements in the list or tuple are trajectory data. The elements in the trajectory are sampling points.
    Returns
    -------
    int
        the best match distance between two trajectory
    """
    m = len(tra1) + 1
    n = len(tra2) + 1
    dp = [[0 for j in range(n)] for i in range(m)]
    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j
    for i in range(1, m):
        for j in range(1, n):
            if tra1[i - 1] == tra2[j - 1]:
                replace_cost = 0
            else:
                replace_cost = 2
            dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i - 1][j - 1] + replace_cost)
    return dp[m - 1][n - 1]


def t_dbscan(trajectory, d=10, y=0.03, score=0.03):
    """
    The outlier detection algorithm T-DBSCAN.

    Parameters
    ----------
    trajectory : list or tuple
        Elements in the list or tuple are trajectory data, and each trajectory
        is a list or tuple. The elements in the trajectory are sampling points,
        which are represented by a binary list or binary tuple. The ID in the
        graph and sampling time of the points are recorded in sequence.
    d : int
        the distance threshold
    y : float
        the outlier threshold
    score : float
        the outlier score

    Returns
    -------
    [TR0_PRE, TR1_PRE, ...] : list
        The prediction sequence of each trajectory.
        0 is normal and 1 is abnormal.

    """
    trs = data_preprocessing(trajectory)
    y_score = []
    num = len(trs)
    core_route = []  # core route

    for i in range(num):
        temp = []
        for item in range(num):
            if i != item:
                if misDistance(trs[i], trs[item]) < d:
                    temp.append(trs[item])  # to get core route, each trajectory has to compare with the others
        if len(temp) >= y * num:
            core_route.append(trs[i])

    for j in range(num):
        t_outlier = []
        for k in range(len(core_route)):
            if misDistance(trs[j],
                           core_route[k]) < d:  # the distance between trs[j] and core_route[k] under the threshold d
                t_outlier.append(trs[j])
        if len(t_outlier) < score * num:
            y_score.append(1)
        else:
            y_score.append(0)
    return y_score
