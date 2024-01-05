"""

date: 22-01-2023 15:39
last edited by: N Surya

"""

import numpy as np
import math

import Global as Gb


g = Gb.Atmos


def atm_cal():
    if g.altitude >= 25000:
        temperature = -131.21 + 0.00299 * g.altitude
        pressure = 2.488 * (((g.altitude + 273.15) / 216.6) ** -11.388)
        density = pressure / (.2869 * (temperature + 273.1))
        g.temperature = temperature
        g.pressure = pressure
        g.density = density

    elif 11000 <= g.altitude <= 25000:
        temperature = -56.46
        pressure = 22.65 * math.exp(1.73 - (0.000157 * g.altitude))
        density = pressure / (.2869 * (temperature + 273.15))
        g.temperature = temperature
        g.pressure = pressure
        g.density = density

    elif g.altitude <= 11000:
        temperature = 15.04 - (0.00649 * g.altitude)
        pressure = 101.29 * (((temperature + 273.15) / 288.08) ** 5.256)
        density = pressure / (.2869 * (temperature + 273.15))
        g.temperature = temperature
        g.pressure = pressure
        g.density = density

    g.a = math.sqrt(1.4 * 287 * (g.temperature + 273.15))
    return g.density


def kinematic_viscosity():
    d = np.divide((g.temperature + 273.15), 273.15) ** 1.5
    b = (384 / (g.temperature + 384.15))
    g.dynamic_viscosity = (1.716 * 10 ** -5) * d * b
    g.kinematic_viscosity = (g.dynamic_viscosity / g.density)
    return g.kinematic_viscosity

