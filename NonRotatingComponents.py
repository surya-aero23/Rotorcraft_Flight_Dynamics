import numpy as np
import math

import ReferenceFrames as Ref
import Global as Gb
import Atmosphere as Atm
import GetInput as Gip


def fuselage_drag(component, V_t, theta):
    drag = 0.5 * Gb.Atmos.density * (V_t ** 2) * component.area
    return [[-drag * math.cos(theta), 0, -drag * math.sin(theta)], [0, 0, 0]]


def empennage_loads(component, mu):
    if component.component_name.lower() == 'horizontal':
        horizontal = True
    else:
        horizontal = False
    rho = Gb.Atmos.density
    v_cg = Ref.vec_to_list(Gb.body_fixed_cg.__dict__['v_net'])
    omega_cg = Ref.vec_to_list(Gb.body_fixed_cg.__dict__['w_net'])
    if horizontal is True:
        R = [-7.325, 0, -0.53]
    else:
        R = [-7.313, 0, -0.452]
    v = np.add(v_cg, np.cross(omega_cg, R))
    v_sqrd = (v[0] ** 2) + (v[1] ** 2) + (v[2] ** 2)
    # v = np.sqrt(float(v_sqrd))
    alpha = 0
    if horizontal is True:
        if mu > 0.05:
            alpha = math.atan(v[2] / v[0])
        theta = math.radians(component.setting_angle) + alpha
    else:
        if mu > 0.05:
            alpha = math.atan(v[1] / v[0])
        theta = math.radians(component.setting_angle) - alpha

    if theta > 0.5:
        lift = 0
    else:
        lift = 0.5 * rho * component.area * component.cl_alpha * theta * v_sqrd

    drag = 0.5 * rho * component.area * v_sqrd * component.cd
    moment = 0.5 * rho * component.area * v_sqrd * component.cm * component.chord
    forces = [0 for _ in range(3)]
    if horizontal is True:
        forces[0] = (lift * math.sin(alpha)) - (drag * math.cos(alpha))
        forces[2] = (-lift * math.cos(alpha)) - (drag * math.sin(alpha))
        moments = [0, moment, 0]
    else:
        forces[0] = (-lift * math.sin(alpha)) - (drag * math.cos(alpha))
        forces[1] = (lift * math.cos(alpha)) - (drag * math.sin(alpha))
        moments = [0, 0, moment]

    rxf = np.cross(R, forces)
    moments = list(np.add(moments, rxf))
    return [forces, moments]


if __name__ == '__main__':
    print('I am GetInput.py')
    directory = 'InputData.xlsx'
    Gip.read_inputs(r'InputData.xlsx')
    Atm.atm_cal()

    n = len(Gb.stabilizer_components)
    for i in range(n):
        print(vars(Gb.stabilizer_components[i]))

    n = len(Gb.fuselage_components)
    for i in range(n):
        print(vars(Gb.fuselage_components[i]))

    n = len(Gb.rotary_components)
    for i in range(n):
        print(vars(Gb.rotary_components[i]))

