"""T-partition and related utilities.
"""
import functools
import math
import numpy as np

from shapely.geometry import Point

MDL_COST_ADVANTAGE = 20


class FineTPartition:
    """
    A one dimensional feature

    A segment has non-zero length and zero area.

    Attributes
    ----------
    start :  list or tuple
        The starting point of the sub-segment is represented by the coordinates of
        the two-dimensional coordinate system (e.g. [x, y] or (x, y)).
    end : list or tuple
        The ending point of the sub-segment is represented by the coordinates of
        the two-dimensional coordinate system (e.g. [x, y] or (x, y)).
    length : float
        The length of the sub-segment.
    seg_distance : list
        The distance between this sub-segment and all other sub-segments.
    density : int
        The sub-segment density, whichis the number of sub-segments within the
        standard deviation from this sub-segment.
    judge : bool
        Mark whether the sub-segment is outlying.
        True: not outlying
        False: outlying

    Example
    -------
      >>> seg = FineTPartition([0, 2], [0, 5])
      >>> print(seg)
      start: [0, 2]; end: [0, 5]; length: 3.0
      >>> seg.start
      [0, 2]
      >>> seg.end
      [0, 5]
      >>> seg.length
      3.0
    """

    def __init__(self, node1, node2, num_trs):
        """
        Parameters
        ----------
        node1 : list or tuple
            The starting point of the sub-segment is represented by the coordinates of
            the two-dimensional coordinate system (e.g. [x, y] or (x, y)).
        node2 : list or tuple
            The starting point of the sub-segment is represented by the coordinates of
            the two-dimensional coordinate system (e.g. [x, y] or (x, y)).
        num_trs : int
            The number of all trajectories.
        """
        self.__start = node1  # start
        self.__end = node2  # end

        self.len_close_seg = [0 for i in range(num_trs)]
        self.count_close_tr = 0
        self.seg_distance = []
        self.density = 0
        self.judge = False
        self.enter = False

    @property
    def start(self):
        """Return the start point ."""
        return self.__start

    @property
    def end(self):
        """Return the end point."""
        return self.__end

    @property
    def length(self):
        """Return the distance from the start point to the end point."""
        p1 = Point(self.start)
        p2 = Point(self.end)
        return p1.distance(p2)

    def __len__(self):
        return self.length

    def __str__(self):
        return "start: {}; end: {}; length: {}".format(self.start, self.end, self.length)


class CoarseTPartition:
    """
    Attributes
    ----------
    start :  list or tuple
        The starting point of the sub-segment is represented by the coordinates of
        the two-dimensional coordinate system (e.g. [x, y] or (x, y)).
    end : list or tuple
        The ending point of the sub-segment is represented by the coordinates of
        the two-dimensional coordinate system (e.g. [x, y] or (x, y)).
    length : float
        The length of the sub-segment.

    Example
    -------
      >>> seg = CoarseTPartition([[0, 2], [0, 5]])
      >>> print(seg)
      start: [0, 2]; end: [0, 5]; length: 3.0
      >>> seg.start
      [0, 2]
      >>> seg.end
      [0, 5]
      >>> seg.length
      3.0
    """

    def __init__(self, nodelist, num_trs=0, notation=False):
        self.__start = nodelist[0]  # start
        self.__end = nodelist[-1]  # end
        self.__notation = notation
        self.subsegment = [FineTPartition(nodelist[i], nodelist[i + 1], num_trs) for i in range(len(nodelist) - 1)]

    @property
    def start(self):
        """Return the start point ."""
        return self.__start

    @property
    def end(self):
        """Return the end point."""
        return self.__end

    @property
    def length(self):
        """Return the distance from the start point to the end point."""
        p1 = Point(self.start)
        p2 = Point(self.end)
        return p1.distance(p2)

    @property
    def max_l_perpendicular(self):
        if self.__notation:
            return max(perpendicular_distance(self, item) for item in self.subsegment)

    @property
    def max_l_len(self):
        if self.__notation:
            return max(item.length for item in self.subsegment)

    @property
    def min_l_len(self):
        if self.__notation:
            return min(item.length for item in self.subsegment)

    @property
    def max_l_angle(self):
        if self.__notation:
            return max(angle(self, item) for item in self.subsegment)

    def __len__(self):
        return self.length

    def __str__(self):
        return "start: {}; end: {}; length: {}".format(self.start, self.end, self.length)


def point_to_line_distance(point, seg):
    """
    Calculate the vertical distance from the point to the sub-segment.

    Parameters
    ----------
    point : list or tuple
        The list or tuple of two elements contains the abscissa
        and ordinate of the point respectively (e.g. [x, y] or (x, y)).
    seg : FineTPartition

    Returns
    -------
    float
        A number is returned as the vertical distance
        from the point to the sub-segment.
    """
    line_point1 = np.array(seg.start)
    line_point2 = np.array(seg.end)
    point1 = np.array(point)
    vec1 = line_point1 - point1
    vec2 = line_point2 - point1
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)

    return distance


@functools.lru_cache(maxsize=512)
def l_perpendicular(seg1, seg2):
    # print(seg1)
    # print(seg2)
    if seg1.length < seg2.length:
        sub_seg1 = seg1
        sub_seg2 = seg2
    else:
        sub_seg1 = seg2
        sub_seg2 = seg1

    pj_s = Point(sub_seg1.start)
    pj_e = Point(sub_seg1.end)
    pi_s = Point(sub_seg2.start)
    pi_e = Point(sub_seg2.end)

    if 0 == sub_seg2.length:  # Both sub-segments are points.
        l_perpendicular_1 = pj_s.distance(pi_s)
        l_perpendicular_2 = pj_e.distance(pi_e)
    else:
        l_perpendicular_1 = point_to_line_distance(sub_seg1.start, sub_seg2)
        l_perpendicular_2 = point_to_line_distance(sub_seg1.end, sub_seg2)

    return l_perpendicular_1, l_perpendicular_2


@functools.lru_cache(maxsize=512)
def perpendicular_distance(seg1, seg2):
    if seg1.length < seg2.length:
        sub_seg1 = seg1
        sub_seg2 = seg2
    else:
        sub_seg1 = seg2
        sub_seg2 = seg1

    pj_s = Point(sub_seg1.start)
    pj_e = Point(sub_seg1.end)
    pi_s = Point(sub_seg2.start)
    pi_e = Point(sub_seg2.end)

    # d⊥(Li, Lj)
    if 0 == sub_seg2.length:  # Both sub-segments are points.
        dis_s_perpendicular = pj_s.distance(pi_s)
        dis_e_perpendicular = pj_e.distance(pi_e)
    else:
        dis_s_perpendicular = point_to_line_distance(sub_seg1.start, sub_seg2)
        dis_e_perpendicular = point_to_line_distance(sub_seg1.end, sub_seg2)

    if 0 == dis_s_perpendicular + dis_e_perpendicular:  # The two sub-segments are on the same line.
        dis_perpendicular = 0
    else:
        dis_perpendicular = (dis_s_perpendicular ** 2 + dis_e_perpendicular ** 2) / (dis_s_perpendicular +
                                                                                     dis_e_perpendicular)

    return dis_perpendicular


@functools.lru_cache(maxsize=512)
def parallel_distance(seg1, seg2):
    if seg1.length < seg2.length:
        sub_seg1 = seg1
        sub_seg2 = seg2
    else:
        sub_seg1 = seg2
        sub_seg2 = seg1

    pj_s = Point(sub_seg1.start)
    pj_e = Point(sub_seg1.end)
    pi_s = Point(sub_seg2.start)
    pi_e = Point(sub_seg2.end)

    if 0 == sub_seg2.length:  # Both sub-segments are points.
        dis_s_perpendicular = pj_s.distance(pi_s)
        dis_e_perpendicular = pj_e.distance(pi_e)
    else:
        dis_s_perpendicular = point_to_line_distance(sub_seg1.start, sub_seg2)
        dis_e_perpendicular = point_to_line_distance(sub_seg1.end, sub_seg2)

    # d∥(Li, Lj)
    dis_ss = pj_s.distance(pi_s)
    dis_se = pj_s.distance(pi_e)
    dis_s_parallel = math.sqrt(min(dis_ss, dis_se) ** 2 - dis_s_perpendicular ** 2)

    dis_es = pj_e.distance(pi_s)
    dis_ee = pj_e.distance(pi_e)
    dis_e_parallel = math.sqrt(min(dis_es, dis_ee) ** 2 - dis_e_perpendicular ** 2)

    dis_parallel = min(dis_s_parallel, dis_e_parallel)

    return dis_parallel


@functools.lru_cache(maxsize=512)
def angle_distance(seg1, seg2):
    if seg1.length < seg2.length:
        sub_seg1 = seg1
        sub_seg2 = seg2
    else:
        sub_seg1 = seg2
        sub_seg2 = seg1

    pj_s = Point(sub_seg1.start)
    pj_e = Point(sub_seg1.end)
    pi_s = Point(sub_seg2.start)
    pi_e = Point(sub_seg2.end)

    if 0 == sub_seg2.length:  # Both sub-segments are points.
        dis_s_perpendicular = pj_s.distance(pi_s)
        dis_e_perpendicular = pj_e.distance(pi_e)
    else:
        dis_s_perpendicular = point_to_line_distance(sub_seg1.start, sub_seg2)
        dis_e_perpendicular = point_to_line_distance(sub_seg1.end, sub_seg2)

    # dθ(Li, Lj)
    dis_angle = abs(dis_s_perpendicular - dis_e_perpendicular)

    return dis_angle


@functools.lru_cache(maxsize=512)
def dist(seg1, seg2, w_perpendicular=1 / 4, w_parallel=1 / 4, w_angle=1 / 2):
    """
    Calculate the distance between two sub-segments.

    Parameters
    ----------
    seg1 : FineTPartition
    seg2 : FineTPartition
    w_perpendicular : float
        The weight of the perpendicular distance.
    w_parallel : float
        The weight of the parallel distance.
    w_angle : float
        The weight of the angle distance.

    The sum of the three distance weights must be 1.

    Returns
    -------
    float
        A number is returned as the distance between two sub-segments.
    """
    if w_perpendicular + w_parallel + w_angle != 1:
        raise ValueError("Weights must sum to 1.")

    if seg1.length < seg2.length:
        sub_seg1 = seg1
        sub_seg2 = seg2
    else:
        sub_seg1 = seg2
        sub_seg2 = seg1

    pj_s = Point(sub_seg1.start)
    pj_e = Point(sub_seg1.end)
    pi_s = Point(sub_seg2.start)
    pi_e = Point(sub_seg2.end)

    # d⊥(Li, Lj)
    if 0 == sub_seg2.length:  # Both sub-segments are points.
        dis_s_perpendicular = pj_s.distance(pi_s)
        dis_e_perpendicular = pj_e.distance(pi_e)
    else:
        dis_s_perpendicular = point_to_line_distance(sub_seg1.start, sub_seg2)
        dis_e_perpendicular = point_to_line_distance(sub_seg1.end, sub_seg2)
    if 0 == dis_s_perpendicular + dis_e_perpendicular:  # The two sub-segments are on the same line.
        dis_perpendicular = 0
    else:
        dis_perpendicular = (dis_s_perpendicular ** 2 + dis_e_perpendicular ** 2) / (dis_s_perpendicular +
                                                                                     dis_e_perpendicular)

    # d∥(Li, Lj)
    dis_ss = pj_s.distance(pi_s)
    dis_se = pj_s.distance(pi_e)
    dis_s_parallel = math.sqrt(min(dis_ss, dis_se) ** 2 - dis_s_perpendicular ** 2)

    dis_es = pj_e.distance(pi_s)
    dis_ee = pj_e.distance(pi_e)
    dis_e_parallel = math.sqrt(min(dis_es, dis_ee) ** 2 - dis_e_perpendicular ** 2)

    dis_parallel = min(dis_s_parallel, dis_e_parallel)

    # dθ(Li, Lj)
    dis_angle = abs(dis_s_perpendicular - dis_e_perpendicular)

    # dist(Li, Lj) = w⊥ * d⊥(Li, Lj) + w∥ * d∥(Li, Lj) + wθ * dθ(Li, Lj)
    weighted_ave_dis = w_perpendicular * dis_perpendicular + w_parallel * dis_parallel + w_angle * dis_angle

    return weighted_ave_dis


@functools.lru_cache(maxsize=512)
def angle(seg1, seg2):
    v1 = np.array(seg1.end) - np.array(seg1.start)
    v2 = np.array(seg2.end) - np.array(seg2.start)
    # print(v1)
    # print(v2)
    if np.all(0 == v1) or np.all(0 == v2):
        return 0
    cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_radians = np.arccos(round(cosine_similarity, 10))  # 弧度制
    # print(angle_radians)
    if math.pi / 2 < angle_radians:
        angle_radians = math.pi - angle_radians

    return angle_radians


def mdl(tra):
    partition = CoarseTPartition(tra)
    l_h = partition.length
    sum_perpendicular = sum([perpendicular_distance(partition, item) for item in partition.subsegment])
    sum_angle = sum([angle_distance(partition, item) for item in partition.subsegment])
    # print('sum_perpendicular', sum_perpendicular)
    # print('sum_angle', sum_angle)
    if 0 == sum_perpendicular or 0 == sum_angle:
        return False
    l_d_h = math.log2(sum_perpendicular) + math.log2(sum_angle)
    cost_par = l_h + l_d_h
    cost_no_par = l_h
    # print(cost_par - cost_no_par)
    return cost_par - cost_no_par > MDL_COST_ADVANTAGE


def approximate_t_partitioning(tra):  # MDL
    cp_id = [0]
    start_index = 0
    length = 1
    while start_index + length <= len(tra) - 1:
        curr_index = start_index + length
        # cost_par, cost_no_par = mdl(tra[start_index: curr_index + 1])
        if mdl(tra[start_index: curr_index + 1]):
            cp_id.append(curr_index - 1)
            start_index = curr_index - 1
            length = 1
        else:
            length += 1
    cp_id.append(len(tra) - 1)

    return cp_id


@functools.lru_cache(maxsize=512)
def get_segment_relation(seg1, seg2):
    x1, y1 = seg1.start
    x2, y2 = seg1.end
    x3, y3 = seg2.start
    x4, y4 = seg2.end

    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])
    # point3
    v3 = np.array([x3 - x1, y3 - y1])
    v4 = np.array([x3 - x2, y3 - y2])
    # point4
    v5 = np.array([x4 - x1, y4 - y1])
    v6 = np.array([x4 - x2, y4 - y2])

    # 快速排斥实验
    if (max(x1, x2) < min(x3, x4)) or (max(x3, x4) < min(x1, x2)) or \
            (max(y1, y2) < min(y3, y4)) or (max(y3, y4) < min(y1, y2)):
        return 'disjoint'

    # 跨立实验
    if np.linalg.det(np.array([v1, v3])) * np.linalg.det(np.array([v1, v5])) <= 0 and \
            np.linalg.det(np.array([v2, v3])) * np.linalg.det(np.array([v2, v4])) <= 0:
        if np.linalg.det(np.array([v1, v2])):
            return 'enclose'
        else:
            return 'overlap'

    return 'disjoint'


@functools.lru_cache(maxsize=512)
def lb_dist(seg1, seg2, w_perpendicular=1 / 4, w_parallel=1 / 4, w_angle=1 / 2):
    if w_perpendicular + w_parallel + w_angle != 1:
        raise ValueError("Weights must sum to 1.")
    tag = get_segment_relation(seg1, seg2)

    lb_perpendicular = min(l_perpendicular(seg1, seg2)) - (seg1.max_l_perpendicular + seg2.max_l_perpendicular)

    lb_parallel = 0 if 'disjoint' != tag else parallel_distance(seg1, seg2)

    sita = angle(seg1, seg2) - seg1.max_l_angle - seg2.max_l_angle
    if sita < 0:
        sita = 0
    elif sita > math.pi / 2:
        sita = math.pi / 2
    lb_angle = min(seg1.min_l_len, seg2.min_l_len) * math.sin(sita)

    weighted_lb_dis = w_perpendicular * lb_perpendicular + w_parallel * lb_parallel + w_angle + lb_angle

    return weighted_lb_dis


@functools.lru_cache(maxsize=512)
def ub_dist(seg1, seg2, w_perpendicular=1 / 4, w_parallel=1 / 4, w_angle=1 / 2):
    if w_perpendicular + w_parallel + w_angle != 1:
        raise ValueError("Weights must sum to 1.")
    tag = get_segment_relation(seg1, seg2)

    ub_perpendicular = max(l_perpendicular(seg1, seg2)) + (seg1.max_l_perpendicular + seg2.max_l_perpendicular)

    if 'enclose' == tag:
        ub_parallel = max(seg1.length, seg2.length)
    elif 'overlap' == tag:
        ub_parallel = seg1.length + seg2.length - parallel_distance(seg1, seg2)
    else:
        ub_parallel = seg1.length + seg2.length + parallel_distance(seg1, seg2)

    sita = angle(seg1, seg2) + seg1.max_l_angle + seg2.max_l_angle
    if sita < 0:
        sita = 0
    elif sita > math.pi / 2:
        sita = math.pi / 2
    ub_angle = min(seg1.max_l_len, seg2.max_l_len) * math.sin(sita)

    weighted_ub_dis = w_perpendicular * ub_perpendicular + w_parallel * ub_parallel + w_angle + ub_angle

    return weighted_ub_dis
