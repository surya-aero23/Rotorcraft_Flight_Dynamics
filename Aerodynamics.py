import numpy as np

import Global as Gb
import ReferenceFrames as Ref
import NumericalMethods as Num


def s_cl(alpha, cl_alpha=5.73):
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


def aerodynamic_load_at_station(component, frame, velocity, theta):
    u_r, u_t, u_p = Ref.vec_to_list(velocity)
    if component.rpm[-1] < 0:
        u_t = -u_t
    phi = np.arctan(float((u_p / u_t)))
    alpha_eff = theta - phi
    c_l = s_cl(alpha_eff, cl_alpha=component.cl_alpha)
    ' Put proper chord, cd0, cm based on section location '
    c = frame.__dict__['chord']
    lift = s_lift(u_p, u_t, c_l, c)
    drag = s_drag(u_p, u_t, c, component.cd0)
    moment = s_moment(u_p, u_t, c, component.cm)
    # moment = 0

    # Loads adjusted for induced inflow angle (Transformed to station co-ords) and direction of blade rotation
    lift_in_station = lift * np.cos(phi) - drag * np.sin(phi)
    drag_in_station = lift * np.sin(phi) + drag * np.cos(phi)
    if component.omega < 0:
        drag_in_station = -drag_in_station

    load_vector = [0, -drag_in_station, lift_in_station]  # Signs given conventionally
    moment_in_station = [moment, 0, 0]
    frame.__dict__.update({'frame_loads': [load_vector, moment_in_station]})
    Gb.section_file.write('\tU_p, U_t: ' + str([u_p, u_t]) + '\ttwist: ' + str(theta) + '\tphi: ' + str(phi) + '\t' + 'Alpha: ' + str(alpha_eff) + '\tLoads: ' + str(load_vector) + '\t' + str(moment_in_station) + '\n')
    # load = Ref.find_loads(frame, frame.__dict__['pv_parent'])
    # Gb.section_file.write(f"{str(load[0])}, {str(load[1])}\t")
    return load_vector, moment_in_station


def inertial_flap_moment_at_frame(component, blade_index, hinge_frame, actual_hinge=True):
    """

    :param component: class object
    :param blade_index: blade no
    :param hinge_frame: hinge or attachment reference frame
    :return: inertial moment
    """

    station_frames = component.stations_ref_frames[blade_index]
    length = len(station_frames)
    acceleration = []
    for i in range(0, length):
        if station_frames[i].__dict__['pv_parent'].name == hinge_frame.name:
            a_net = station_frames[i].__dict__['a_net']  # use update code and store the acceleration beforehand
            # inertial_acc = -component.rho_body * ((station_frames[i].__dict__['position_vector']).cross(Ref.transform_vector(a_net, station_frames[i], hinge_frame)))
            inertial_acc = -component.rho_body * (Ref.cross_pdt(station_frames[i].__dict__['position_vector'], Ref.transform_vector(a_net, station_frames[i], hinge_frame)))
            acceleration.append(inertial_acc.dot(hinge_frame.y))

    lower_lim = (hinge_frame.__dict__['position_vector']).dot(hinge_frame.x)
    upper_lim = component.radius
    inertial_moment = Num.simpsons_rule(acceleration, lower_lim, upper_lim)

    if actual_hinge is True:
        return inertial_moment
    else:
        inertial_moment = Ref.transform_vector(inertial_moment * hinge_frame.y, hinge_frame, actual_hinge)
        return Ref.vec_to_list(inertial_moment)[1]


def beta_double_dot(beta_list, t, component, blade_index, aerodynamic_moment, flap_hinge_frame, intermediate_hinge_frame=None):
    beta = beta_list
    Ref.orient_azimuth_with_psi(component.rotary_ref_frames[0], component.rotary_ref_frames[blade_index+1][0], component, blade_index + 1, t * component.omega)
    Ref.orient_with_axis(flap_hinge_frame, flap_hinge_frame.__dict__['pv_parent'], 'y', beta[0])
    flap_hinge_frame.__dict__['w_frame'] = beta[1] * flap_hinge_frame.y

    # velocity_update_between_hinges(component, blade_index + 1)
    flap_hinge_frame.__dict__['alpha_frame'] = 0
    vel_update_for_flap(component, blade_index)
    if intermediate_hinge_frame is not None:
        inertial_moment_1 = inertial_flap_moment_at_frame(component, blade_index, intermediate_hinge_frame, actual_hinge=flap_hinge_frame)
    else:
        inertial_moment_1 = inertial_flap_moment_at_frame(component, blade_index, flap_hinge_frame)

    flap_hinge_frame.__dict__['alpha_frame'] = 1 * flap_hinge_frame.y
    vel_update_for_flap(component, blade_index)
    if intermediate_hinge_frame is not None:
        inertial_moment_2 = inertial_flap_moment_at_frame(component, blade_index, intermediate_hinge_frame, actual_hinge=flap_hinge_frame)
    else:
        inertial_moment_2 = inertial_flap_moment_at_frame(component, blade_index, flap_hinge_frame)
    beta_dob_dot = (-aerodynamic_moment - inertial_moment_1 - (component.k_beta * beta[0])) / (inertial_moment_2 - inertial_moment_1)
    return np.array((beta[1], beta_dob_dot))


def vel_update_for_flap(component, blade_index):
    for i in range(1, 4):
        Ref.update_net_vel(component.rotary_ref_frames[blade_index + 1][i], acceleration=True)
    for i in range(0, Gb.RotatingComponent.no_of_stations):
        Ref.update_net_vel(component.stations_ref_frames[blade_index][i], acceleration=True)
    return 0


if __name__ == '__main__':
    print('I am Aerodynamics.py')
