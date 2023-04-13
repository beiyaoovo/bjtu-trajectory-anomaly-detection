"""Trajectory sub-segments and related utilities
"""

import math
import numpy as np

from shapely.geometry import Point


class SubSegment:
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
    len_close_seg : list
        The total length of adjacent sub-segments from each trajectory
        (excluding the track where the sub-segment is located).
    count_close_tr : int
        The number of adjacent trajectories.
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
      >>> seg = SubSegment([0, 2], [0, 5])
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
        """
        self.__start = node1  # start
        self.__end = node2  # end

        self.len_close_seg = [0 for i in range(num_trs)]
        self.count_close_tr = 0
        self.seg_distance = []
        self.density = 0
        self.judge = False

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


def distance_point_to_line(point, seg):
    """
    Calculate the vertical distance from the point to the sub-segment.

    Parameters
    ----------
    point : list or tuple
        The list or tuple of two elements contains the abscissa
        and ordinate of the point respectively (e.g. [x, y] or (x, y)).
    seg : SubSegment

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


def dist(seg1, seg2, w_perpendicular=1 / 4, w_parallel=1 / 4, w_angle=1 / 2):
    """
    Calculate the distance between two sub-segments.

    Parameters
    ----------
    seg1 : SubSegment
    seg2 : SubSegment
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
        exit(1)
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
        dis_s_perpendicular = distance_point_to_line(sub_seg1.start, sub_seg2)
        dis_e_perpendicular = distance_point_to_line(sub_seg1.end, sub_seg2)
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
