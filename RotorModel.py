import numpy as np
from operator import sub
import Global as Gb
import NumericalMethods as Num
import ReferenceFrames as Ref


def inflow_calculation(component, load_coefficients, psi, inflow_type, initial_inflow=None):      # Using Newton-Raphson method
    v_hub = Ref.vec_to_list(component.rotary_ref_frames[0].__dict__['v_net'])

    if isinstance(load_coefficients, list) is not True:
        c_t = load_coefficients
        load_coefficients = [c_t, 0, 0]
    else:
        c_t = load_coefficients[0]

    if v_hub is None:
        v_hub = [0, 0, 0]

    mu = np.sqrt(v_hub[0] ** 2 + v_hub[1] ** 2) / (component.radius * component.omega)
    w_nondim = v_hub[2] / (component.radius * component.omega)

    # def find_uniform_inflow(lamda_in_list):
    #     lamda = lamda_in_list
    #     return lamda - (c_t / (2 * (np.sqrt(mu ** 2 + (-w_nondim + lamda) ** 2))))
    #
    # uniform_inflow = float(Num.simplified_newton_raphson(find_uniform_inflow, initial_inflow[0]))
    def find_uniform_inflow(mu, w_nondim, c_t, initial_inflow):
        lamda_new = np.zeros(100)

        i = 0
        while i < 1000:
            lamda_new[i] = initial_inflow
            f = lamda_new[i] - (c_t / (2 * pow(mu ** 2 + ((-w_nondim + lamda_new[i]) ** 2), 0.5)))
            dlamdai = lamda_new[i] + (0.05 * lamda_new[i])
            df1 = dlamdai - (c_t / (2 * pow(mu ** 2 + ((-w_nondim + dlamdai) ** 2), 0.5)))
            df = (df1 - f) / (0.05 * lamda_new[i])
            lamda_new[i + 1] = lamda_new[i] - (f / df)
            initial_inflow = lamda_new[i + 1]

            if abs(lamda_new[i + 1] - lamda_new[i]) < 1e-12:
                lambda_i = lamda_new[i + 1]
                return lambda_i
            i += 1

    uniform_inflow = find_uniform_inflow(mu, w_nondim, c_t, initial_inflow)

    match inflow_type:
        case 1:
            'lamda 0, lamda 1s, lamda 1c'
            return [uniform_inflow, 0, 0]

        case 2:
            if mu == 0:
                lambda_i = uniform_inflow
                lambda_0 = lambda_i
                lambda_1c = 0
                lambda_1s = 0

            else:
                lambda_i = uniform_inflow
                lambda_0 = -w_nondim + lambda_i
                kai = np.arctan(mu / lambda_0)
                lambda_1c = lambda_i * 4.0 * ((1 - 1.8 * pow(mu, 2)) * (1 / np.sin(kai)) - (1 / np.tan(kai))) / 3.0
                lambda_1s = -lambda_i * 2.0 * mu
            'lamda 0, k_x --> 1s, k_y --> 1c'
            return [lambda_0, lambda_1s, lambda_1c]

        case 3:
            if initial_inflow is None:
                initial_inflow = [uniform_inflow, 0, 0]
            matrix_el, matrix_m, matrix_v = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))

            def initiate_dynamic_inflow(component, velocity, load_coefficients):
                u, v, w = velocity
                if mu != 0:
                    lamda_i = uniform_inflow + v
                else:
                    lamda_i = uniform_inflow

                chi = np.arctan(mu / lamda_i)
                v_t = np.sqrt(mu ** 2 + lamda_i ** 2)
                # v_r = (mu ** 2 + uniform_inflow * (uniform_inflow + lamda_i)) / v_t
                v_r = 0

                matrix_el[0][0], matrix_el[0][1], matrix_el[0][2] = [0.5, 0, (15 * np.pi / 64) * np.tan(chi / 2)]
                matrix_el[1][0], matrix_el[1][1], matrix_el[1][2] = [0, -4 / (1 + np.cos(chi)), 0]
                matrix_el[2][0], matrix_el[2][1], matrix_el[2][2] = [(15 * np.pi / 64) * np.tan(chi / 2), 0,
                                                                     (-4 * np.cos(chi) / (1 + np.cos(chi)))]

                matrix_m[0][0], matrix_m[0][1], matrix_m[0][2] = [8 / (3 * np.pi), 0, 0]
                matrix_m[1][0], matrix_m[1][1], matrix_m[1][2] = [0, -16 / (45 * np.pi), 0]
                matrix_m[2][0], matrix_m[2][1], matrix_m[2][2] = [0, 0, -16 / (45 * np.pi)]

                matrix_v[0][0], matrix_v[0][1], matrix_v[0][2] = [v_t, 0, 0]
                matrix_v[1][0], matrix_v[1][1], matrix_v[1][2] = [0, v_r, 0]
                matrix_v[2][0], matrix_v[2][1], matrix_v[2][2] = [0, 0, v_r]

                return 0

            def dynamic_inflow(inflow, t, load_coefficients):
                inflow = [float(x) for x in inflow]
                L_inv = np.linalg.inv(matrix_el)
                M_inv = np.linalg.inv(matrix_m)
                VL_inv = np.matmul(np.matmul(matrix_v, L_inv), inflow)
                lamda_dot = list(map(sub, load_coefficients, VL_inv))
                lamda_dot = np.matmul(M_inv, lamda_dot)
                return lamda_dot

            initiate_dynamic_inflow(component, v_hub, load_coefficients)
            # t = psi / component.component.omega
            # dt = Gb.RotatingComponent.d_psi / component.component.omega
            t = psi
            dt = Gb.RotatingComponent.d_psi
            inflow = Num.runge_kutta_solver(dynamic_inflow, initial_inflow, t, t + dt, dt, [load_coefficients])
            'lamda 0, lamda 1s, lamda 1c'
            return inflow


def final_inflow_value(inflow, w, r_bar=0.75, psi=0, inflow_type=1, write=True):
    # u, v, w = ReferenceFrames.vec_to_list(component.rotary_ref_frames[0].__dict__['v_net'])
    if inflow_type == 1:
        'Uniform Inflow'
        return inflow[0]

    if inflow_type == 2:
        'Drees Inflow'
        'lamda 0, k_x --> 1s, k_y --> 1c'
        lamda_0, lamda_1s, lamda_1c = inflow[0], inflow[1] * r_bar * np.sin(psi), inflow[2] * r_bar * np.cos(psi)
        # inflow = inflow[0] + (inflow[1] * r_bar * np.sin(psi)) + (inflow[2] * r_bar * np.cos(psi))
        return lamda_0 + lamda_1s + lamda_1c

    if inflow_type == 3:
        'Dynamic Inflow'
        'lamda 0, lamda 1s, lamda 1c'
        inflow = w + inflow[0] + (inflow[1] * r_bar * np.sin(psi)) + (inflow[2] * r_bar * np.cos(psi))
        return inflow
    else:
        raise ValueError('Invalid inflow type')
