import numpy as np
from time import time

import Global as Gb
import GetInput as Gip
import ReferenceFrames as Ref
import NumericalMethods as Num
import Atmosphere as Atm
import RotorLoadsModule as Rotor
from NonRotatingComponents import fuselage_drag, empennage_loads

Trim_guess = [0.0174532925 * 8.21645, 0.0174532925 * 0.827019 * 1, 0.0174532925 * -0.255768 * 1, 0.0174532925 * 9.92829, 0.0174532925 * 0.3728, 0.0174532925 * -2.75]
# Trim_guess = [0.12041579400261745, 0.0016960157822774294, -0.009617649203876762, 0.1616244490896768, -0.005891217329634712, -0.042270676528904524]


Atm.atm_cal()
Gip.read_inputs(r'InputData.xlsx')
v_cg = [0, 0, 0]
Ref.set_angles([0, 0, 0, 0, 0, 0])
Ref.create_cg_frame(v_cg)
Ref.create_rotating_component(Gb.rotary_components)


def flight_eq_motion(control_guesses, components, main=True, tail=True, save_loads=False):
    control_guesses = list(map(float, control_guesses))
    print('\nInputs:\t', list(map(np.degrees, control_guesses)))
    # print(list(map(np.degrees, control_guesses)))
    g = Gb.g

    def f_x(translational_vel, angular_vel, euler_angle, F_x, m):
        u, v, w = translational_vel
        p, q, r = angular_vel
        u_dot = -(w * q - v * r) + (F_x / m) - g * np.sin(euler_angle[0])
        return float(u_dot)

    def f_y(translational_vel, angular_vel, euler_angle, F_y, m):
        u, v, w = translational_vel
        p, q, r = angular_vel
        v_dot = -(u * r - w * p) + (F_y / m) + g * np.cos(euler_angle[0]) * np.sin(euler_angle[1])
        return float(v_dot)

    def f_z(translational_vel, angular_vel, euler_angle, F_z, m):
        u, v, w = translational_vel
        p, q, r = angular_vel
        w_dot = -(v * p - u * q) + (F_z / m) + g * np.cos(euler_angle[0]) * np.cos(euler_angle[1])
        return float(w_dot)

    def m_x(I_xx, I_yy, I_zz, I_xz, angular_vel, moments):
        L, M, N = moments
        p, q, r = angular_vel
        p_dot = (q * ((r * (I_zz * (I_yy - I_zz)) - I_xz ** 2) + p * (I_xz * (I_xx - I_yy + I_zz))) + I_zz * L + I_xz * N) / (I_xx * I_zz - (I_xz ** 2))
        return float(p_dot)

    def m_y(I_xx, I_yy, I_zz, I_xz, angular_vel, moments):
        L, M, N = moments
        p, q, r = angular_vel
        q_dot = ((I_zz - I_xx) * r * p + I_xz * (r ** 2 - p ** 2) + M) / I_yy
        return float(q_dot)

    def m_z(I_xx, I_yy, I_zz, I_xz, angular_vel, moments):
        L, M, N = moments
        p, q, r = angular_vel
        r_dot = ((q * (r * (I_xz * (I_yy - I_zz - I_xz)) + p * (I_xx * (I_xx - I_yy) + I_xz ** 2))) + I_xz * L + I_xx * N) / (I_xx * I_zz - I_xz ** 2)
        return float(r_dot)

    class Trim:  # will be from input and will be stored in Global
        # Euler Angles
        theta = 0  # pitch angle
        phi = 0  # roll angle
        psi = 0  # yaw angle

        # angular velocities
        p = 0  # pitch rate
        q = 0  # roll rate
        r = 0  # yaw rate

        # translational velocities
        u = 0  # along x
        v = 0  # along y
        w = 0  # along w

        # Rate of euler angle change
        theta_dot = 0
        phi_dot = 0
        psi_dot = 0

        V_t = Gb.V_t  # flight velocity

        flight_path_angle = 0
        pitch_attitude = theta
        track_angle = 0
        heading_angle = 0

        alpha = flight_path_angle - pitch_attitude  # gamma - pitch_attitude
        beta = 0  # track_angle - heading_angle

        I_xx = 5000
        I_yy = 20000
        I_zz = 16700
        I_xz = 3700

    # Updating theta and phi with guess
    Trim.theta = control_guesses[4]
    Trim.phi = control_guesses[5]

    cos_theta, sin_theta = np.cos(Trim.theta), np.sin(Trim.theta)
    cos_gamma, sin_gamma = np.cos(Trim.flight_path_angle), np.sin(Trim.flight_path_angle)
    cos_kai, sin_kai = np.cos(Trim.track_angle), np.sin(Trim.track_angle)
    cos_phi, sin_phi = np.cos(Trim.phi), np.sin(Trim.phi)
    u_ea = Trim.V_t * ((cos_theta * cos_gamma * cos_kai) - sin_theta * sin_gamma)
    v_ea = Trim.V_t * (
                (cos_phi * cos_gamma * sin_kai) + sin_phi * ((sin_theta * cos_gamma * cos_kai) + cos_theta * sin_gamma))
    w_ea = Trim.V_t * ((-sin_phi * cos_gamma * sin_kai) + cos_phi * (
                (sin_theta * cos_gamma * cos_kai) + cos_theta * sin_gamma))
    V_body = Ref.vectorize_list([u_ea, v_ea, w_ea], Gb.body_fixed_cg)
    Gb.body_fixed_cg.__dict__.update({'v_naught': V_body, 'v_net': V_body})

    # Rate of euler angle change
    Trim.phi_dot = Trim.p + np.tan(Trim.theta) * (Trim.q * np.sin(Trim.phi) + Trim.r * np.cos(Trim.phi))
    Trim.theta_dot = Trim.q * np.cos(Trim.phi) - Trim.r * np.sin(Trim.phi)
    Trim.psi_dot = (Trim.q * np.sin(Trim.phi) + Trim.r * np.cos(Trim.phi)) / (np.cos(Trim.theta))

    angular_velocity = [Trim.p, Trim.q, Trim.r]
    angular_velocity = Ref.vectorize_list(angular_velocity, Gb.body_fixed_cg)
    Gb.body_fixed_cg.__dict__.update({'w_frame': angular_velocity, 'w_net': angular_velocity})
    Gb.trim_file.write(f'\nBody translation velocity: {Ref.vec_to_list(V_body)}\tBody angular velocity: {Ref.vec_to_list(angular_velocity)}\n')

    # Force calculations
    [F_x, F_y, F_z], [M_x, M_y, M_z] = Rotor.find_loads_at_cg(control_guesses[0:4], others=[Trim.V_t, Trim.theta, Gb.mu], main=main, tail=tail, save_loads=save_loads)
    fuselage = [[0, 0, 0], [0, 0, 0]]
    empennage = [[0, 0, 0], [0, 0, 0]]
    if isinstance(Gb.fuselage_components, list):
        for i in range(len(Gb.fuselage_components)):
            load = fuselage_drag(Gb.fuselage_components[i], Trim.V_t, Trim.theta)
            fuselage = np.add(load, fuselage)
            Gb.components_file.write(f'\n{Gb.fuselage_components[i].component_name} loads about CG:\t{str(load)}\n')

    if isinstance(Gb.stabilizer_components, list):
        for i in range(len(Gb.stabilizer_components)):
            load = empennage_loads(Gb.stabilizer_components[i], Gb.mu)
            empennage = np.add(load, empennage)
            Gb.components_file.write(f'\n{Gb.stabilizer_components[i].component_name} loads about CG:\t{str(load)}\n')

    total_loads = np.add(fuselage, empennage)
    [F_x, F_y, F_z], [M_x, M_y, M_z] = np.add([[F_x, F_y, F_z], [M_x, M_y, M_z]], total_loads)

    euler_angle = [Trim.theta, Trim.phi, Trim.psi]
    u_dot = f_x([u_ea, v_ea, w_ea], Ref.vec_to_list(angular_velocity), euler_angle, F_x, Gb.vehicle_parameters[0].mass)
    v_dot = f_y([u_ea, v_ea, w_ea], Ref.vec_to_list(angular_velocity), euler_angle, F_y, Gb.vehicle_parameters[0].mass)
    w_dot = f_z([u_ea, v_ea, w_ea], Ref.vec_to_list(angular_velocity), euler_angle, F_z, Gb.vehicle_parameters[0].mass)
    p_dot = m_x(Trim.I_xx, Trim.I_yy, Trim.I_zz, Trim.I_xz, Ref.vec_to_list(angular_velocity), [M_x, M_y, M_z])
    q_dot = m_y(Trim.I_xx, Trim.I_yy, Trim.I_zz, Trim.I_xz, Ref.vec_to_list(angular_velocity), [M_x, M_y, M_z])
    r_dot = m_z(Trim.I_xx, Trim.I_yy, Trim.I_zz, Trim.I_xz, Ref.vec_to_list(angular_velocity), [M_x, M_y, M_z])
    return u_dot, v_dot, w_dot, p_dot, q_dot, r_dot


start = time()
sol = Num.newton_raphson_root_finder(flight_eq_motion, Trim_guess, [Gb.fixed_components])
Gb.trim_file.write(f'\nFinal Trim Inputs (radians):\t{str(sol)}')
Gb.trim_file.write(f'\nTime to perform Trim:\t{str((time() - start) / 60)} minutes')
print(f'Final Trim Inputs:\t{list(map(np.degrees, sol))}')
Gb.trim_file.close()
Gb.components_file.close()
Gb.section_file.close()

