import math
from operator import itemgetter
import numpy as np

import os
import time

# Global variables and class declaration

# List initiation
ref_frames, station_data, llt_pre_var = [], [], []
rotary_components, fixed_components, vehicle_parameters, stabilizer_components, fuselage_components = [], [], [], [], []
body_fixed_cg = []

# Constants
g, PI = 9.81, math.pi

# Pilot Inputs
# theta_0, theta_1c, theta_1s = [0.0872664626 * 8 / 5, 0.0872664626 * 0 / 5, 0.0872664626 * 0 / 5]
# theta_0, theta_1c, theta_1s = [0, 0, 0]


class Atmos:
    altitude = 0     # in meters
    temperature = 0
    pressure = 0
    density = 0
    dynamic_viscosity = 0
    kinematic_viscosity = 0
    a = 0  # speed of sound


class RotatingComponent:
    def __init__(self, component_name, euler, locations, blade_no, rpm, radius, omega_rf, rho_body, cd0, cm, cl_alpha, stations):        # Add cd0 and other things
        self.component_name = component_name
        self.euler = euler
        self.locations = locations
        self.blade_no = blade_no
        self.rpm = rpm
        self.omega = math.sqrt(rpm[0] ** 2 + rpm[1] ** 2 + rpm[2] ** 2)
        self.radius = radius
        self.omega_rf = omega_rf
        self.k_beta = 0
        self.rho_body = rho_body
        self.cd0 = cd0
        self.cm = cm
        self.stations = stations
        self.cl_alpha = cl_alpha
        self.stations_ref_frames = []
        self.rotary_ref_frames = []
        self.station_data = []

    def get_kbeta(self):

        component_name, omega_rf, omega, radius, rho_b, hinge_offset = \
            itemgetter('component_name', 'omega_rf', 'omega', 'radius', 'rho_body', 'locations')(self.__dict__)

        hinge_offset = hinge_offset[1][0]
        if hinge_offset == 0:
            return 0
        else:
            omega_rf = omega_rf / omega
            # hinge_offset = 0.607
            mass_moi = rho_b * ((radius ** 3) / 3)
            kbeta = np.zeros(1000)
            kbeta_init = 100000
            i = 0
            while abs(kbeta[i + 1] - kbeta[i]) < 1e-12 and i > 0:
                f, df1, df, dKbeta = 0, 0, 0, 0
                kbeta[i] = kbeta_init
                f = (omega_rf ** 2) - (1 + (3 * hinge_offset / (2 * (radius - hinge_offset)))) + (kbeta[i]) / (
                            mass_moi * (omega ** 2))
                dkbeta = kbeta[i] + (0.05 * kbeta[i])
                df1 = (omega_rf ** 2) - (1 + (3 * hinge_offset / (2 * (radius - hinge_offset)))) + dkbeta / (
                            mass_moi * (omega ** 2))
                df = (df1 - f) / ((0.05 * kbeta[i]))
                kbeta[i + 1] = kbeta[i] - (f / df)
                kbeta_init = kbeta[i + 1]

                i += 1

            if abs(kbeta[i + 1] - kbeta[i]) < 1e-12:
                kbeta = kbeta[i + 1]
                return kbeta
            else:
                print(f'k_beta did not converge for {self.component_name}')
                return None

    no_of_stations = 19
    d_psi = (PI / 180) * 5


class FixedComponent:
    class GlobalVariables:
        def __init__(self, name, location, span, area, y_st, c_st, a_a_st, a_g_st, sweep_st, hedral_st, con, n):
            self.name = name
            self.location = location
            self.span = span
            self.area = area
            self.y_st = y_st
            self.c_st = c_st
            self.a_a_st = a_a_st
            self.a_g_st = a_g_st
            self.sweep_st = sweep_st
            self.hedral_st = hedral_st
            self.con = con
            self.n = n

    class PreVariables:
        def __init__(self, theta, theta_st, c_y, y, a_g_y, a_a_y):
            self.theta = theta
            self.theta_st = theta_st
            self.c_y = c_y
            self.y = y
            self.a_g_y = a_g_y
            self.a_a_y = a_a_y

    class Controls:
        def __init__(self, con_location, cf, theta_def):
            self.con_location = con_location
            self.cf = cf
            self.theta_def = theta_def


class VehicleParameter:
    def __init__(self, mass, I_xx, I_yy, I_zz, I_xz):
        self.mass = mass
        self.I_xx = I_xx
        self.I_yy = I_yy
        self.I_zz = I_zz
        self.I_xz = I_xz


class Stabilizers:
    def __init__(self, component_name, position, area, chord, cd, cm, setting_angle, cl_alpha):
        self.component_name = component_name
        self.position = position
        self.area = area
        self.chord = chord
        self.cd = cd
        self.cm = cm
        self.setting_angle = setting_angle
        self.cl_alpha = cl_alpha


class Fuselage:
    def __init__(self, component_name, area):
        self.component_name = component_name
        self.area = area


# Log file preparations
mu = 0.00
V_t = mu * 32.88 * 6.6
print(f'mu = {mu}\tVt = {V_t}')
case = f"{time.strftime('''%d.%m.%Y - %H h %M m %S s''')} - New_ThetadotUpdate_mu{str(mu)}"
# folder_name = 'TrimCases'
# case = f"{time.strftime('''%d.%m.%Y - %H h %M m %S s''')} - TRIALmu{mu}"
# folder_name = 'TrialTrimCases'
folder_name = 'TestCases'
newpath = rf'TestFile\{folder_name}\{case}'
os.makedirs(newpath)
trim_file = rf'TestFile\{folder_name}\{case}\trim_data_mu{str(mu)}.txt'
trim_file = open(trim_file, 'w')
trim_file.write(f'\nmu: {str(mu)}\nV_TAS: {str(V_t)}\n')
section_file = rf'TestFile\{folder_name}\{case}\section_data_mu{str(mu)}.txt'
section_file = open(section_file, 'w')
components_file = rf'TestFile\{folder_name}\{case}\components_data_mu{str(mu)}.txt'
components_file = open(components_file, 'w')
output_file = rf'TestFile\{folder_name}\{case}\console_copy_mu{str(mu)}.txt'
output_file = open(output_file, 'w')
inflow_file = rf'TestFile\{folder_name}\{case}\inflow_mu{str(mu)}.txt'
inflow_file = open(inflow_file, 'w')
inflow_file.write('Psi, c_t, T, initial inflow, inflow\n')
# inflow_file.write('Psi, lamda_0, lamda_1s, lamda_1c\n')

