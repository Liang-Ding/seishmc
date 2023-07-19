# -------------------------------------------------------------------
# Tools and utilities
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import numpy as np


def wrap_strike(angle_in_deg):
    '''Wrap strike angle to (0, 360)'''
    if angle_in_deg < 0:
        return np.mod(angle_in_deg+360., 360.)
    if angle_in_deg > 360.:
        return np.mod(angle_in_deg, 360.)
    return angle_in_deg


def wrap_dip(angle_in_deg):
    '''Wrap dip angle to (0, 90)'''
    if angle_in_deg > 90:
        return 90. - np.mod(angle_in_deg, 90)
    if angle_in_deg < 0:
        return np.fabs(angle_in_deg)
    return angle_in_deg


def wrap_rake(angle_in_deg):
    '''Wraps slip angle to (-90, 90)'''
    if angle_in_deg > 90:
        return 90. - np.mod(angle_in_deg, 90)
    if angle_in_deg < -90:
        return -90. - np.mod(angle_in_deg, -90)
    return angle_in_deg


def wrap_colatitude(angle_in_deg):
    '''Wrap colatitude to (0, 180)'''
    if angle_in_deg > 180:
        return 180. - np.mod(angle_in_deg, 180)
    if angle_in_deg < 0:
        return np.fabs(angle_in_deg)
    return angle_in_deg


def wrap_lunelongitude(angle_in_deg):
    '''Wrap colatitude to (-30, 30)'''

    if angle_in_deg > 30:
        return 30. - np.mod(angle_in_deg, 30)
    if angle_in_deg < -30:
        return -30. - np.mod(angle_in_deg, -30)
    return angle_in_deg

