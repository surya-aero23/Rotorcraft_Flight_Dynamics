import collections.abc
import pandas as pd
from math import isnan

import Global as Gb


def read_inputs(file_name):
    df = pd.read_excel(file_name, sheet_name=0, header=None)
    df = df.dropna(thresh=3)
    df = df.reset_index()
    pd.options.display.width = 0
    for i in range(1, len(df[0])):
        if isnan(float(df[0][i])) is False:
            num = df[0][i]
            if num == 0:
                mass = df[2][1]
                I_xx, I_yy, I_zz, I_xz = df[2][2:6]
                vehicle_parameter = Gb.VehicleParameter(mass, I_xx, I_yy, I_zz, I_xz)
                Gb.vehicle_parameters.append(vehicle_parameter)
            if num > 0:
                if df[2][i] == 'Rotating':
                    locations = [[float(x) for x in df[2][i + j + 3].split(',')] for j in range(4)]
                    euler = [float(x) for x in df[2][i + 2].split(',')]
                    rpm = [float(x) for x in df[2][i + 8].split(',')]
                    radius = float(df[2][i + 9])
                    omega_rf = float(df[2][i + 10])
                    rho_body = float(df[2][i + 11])
                    cd0 = float(df[2][i + 12])
                    cm = float(df[2][i + 13])
                    cl_alpha = float(df[2][i + 14])
                    j = i + 1
                    check = True
                    sections = []
                    while check is True:
                        if df[1][j] == 'section_location':
                            h = j + 1
                            while df[1][h] != 'End of station data':
                                sections.append([float(df[k][h]) for k in range(1, 5)])
                                h += 1
                            check = False
                        j += 1
                    component = Gb.RotatingComponent(df[2][i + 1], euler, locations, int(df[2][i + 7]), rpm, radius, omega_rf, rho_body, cd0, cm, cl_alpha, sections)
                    # component.k_beta = component.get_kbeta()
                    Gb.rotary_components.append(component)

                elif df[2][i] == 'Fixed':
                    a = Gb.FixedComponent()
                    location = [float(x) for x in df[2][i + 2].split(',')]
                    span = df[2][i + 3]
                    area = df[2][i + 4]
                    y_st = []
                    c_st = []
                    a_a_st = []
                    a_g_st = []
                    sweep_st = []
                    hedral_st = []
                    par = [y_st, c_st, a_a_st, a_g_st, sweep_st, hedral_st]
                    for k in range(1, len(par) + 1):
                        e = 0
                        t = 0
                        while e == 0:
                            t = t + 1
                            if df[k][i + t + 5] == 'End':
                                e = 1
                            else:
                                if isinstance(df[k][i + t + 5], int) or isinstance(df[k][i + t + 5], float):
                                    u = df[k][i + t + 5]
                                    par[k - 1].append(u)
                                    f = t
                                else:
                                    e = 1
                    con = []
                    m = 0
                    n = 0
                    r = i + f + 7
                    while m == 0:
                        con_loc = []
                        cf = 0
                        theta_def = 0
                        par_con = [con_loc, cf, theta_def]
                        n = n + 1
                        for k in range(1, 4):
                            if df[k][r + n] == 'End':
                                m = 1
                            else:
                                if isinstance(df[k][r + n], collections.abc.Sequence) or isinstance(df[k][r + n],
                                                                                                    float) or \
                                        isinstance(df[k][r + n], int):
                                    if k == 1:
                                        loc = [float(x) for x in df[k][r + n].split(',')]
                                        par_con[k - 1] = loc
                                    else:
                                        u = df[k][r + n]
                                        par_con[k - 1] = u
                                else:
                                    m = 1
                        if par_con[1] == 0 and par_con[2] == 0:
                            pass
                        else:
                            y = a.Controls(par_con[0], par_con[1], par_con[2])
                            con.append(y)

                    n = 36
                    component = a.GlobalVariables(df[2][i + 1], location, span, area, par[0], par[1], par[2], par[3],
                                                  par[4], par[5], con, n)
                    Gb.fixed_components.append(component)

                if df[2][i] == 'Stabilizer':
                    position = [float(x) for x in df[2][i + 2].split(',')]
                    s = float(df[2][i + 3])
                    c = float(df[2][i + 4])
                    cd = float(df[2][i + 5])
                    cm = float(df[2][i + 6])
                    a = float(df[2][i + 7])
                    cl_alpha = float(df[2][i + 8])
                    component = Gb.Stabilizers(df[2][i + 1], position, s, c, cd, cm, a, cl_alpha)
                    Gb.stabilizer_components.append(component)

                if df[2][i] == 'Fuselage':
                    area = float(df[2][i + 2])
                    component = Gb.Fuselage(df[2][i + 1], area)
                    Gb.fuselage_components.append(component)
    return df


if __name__ == '__main__':
    print('I am GetInput.py')
    directory = 'InputData.xlsx'
    df = read_inputs(directory)

    n = len(Gb.stabilizer_components)
    for i in range(n):
        print(vars(Gb.stabilizer_components[i]))

    n = len(Gb.fuselage_components)
    for i in range(n):
        print(vars(Gb.fuselage_components[i]))
