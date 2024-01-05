import math

import Global as Gb
PI = math.pi


def s_cl(alpha):
    alpha_0 = 0  # Zero lift AoA
    # a = 2 * PI
    a = 5.73   # Get sectional lift curve slope
    return a * (alpha - alpha_0)


def s_lift(up, ut, cl, chord):
    return 0.5 * Gb.Atmos.density * chord * ((up ** 2) + (ut ** 2)) * cl


def s_drag(up, ut, chord, cd):
    return 0.5 * Gb.Atmos.density * chord * ((up ** 2) + (ut ** 2)) * cd


def s_moment(up, ut, chord, cm):
    return 0.5 * Gb.Atmos.density * (chord ** 2) * (up ** 2 + ut ** 2) * cm

