import re
import time

import sympy as sm
from sympy import cos, sin, Matrix
import numpy as np
from operator import itemgetter

import sympy.vector
from sympy.physics import vector as vec

import Global as Gb
from NumericalMethods import interpolate

PI = np.pi
zero_vector_list = [0, 0, 0]


def set_angles(angles=None):
    if angles is None:
        angles = [0, 0, 0, 0, 0, 0]
    global shaft_angle, t, theta_i, beta_k, lag_k, torsion_k
    shaft_angle, t, theta_i, beta_k, lag_k, torsion_k = angles
    return 0


def vectorize_list(vector, frame, n=1):
    """

    :param vector: vector as a list in the given reference frame
    :param frame: frame object in which vector is defined
    :param n: number of vectors in the list, 1 by default
    :return: sympy.physics.vector.Vector object / objects
    """
    if n == 1:
        v_i = vector[0] * frame.x + vector[1] * frame.y + vector[2] * frame.z
        return v_i

    if n > 1:
        converted_vectors = []
        for i in range(0, n):
            v_i = vector[i][0] * frame.x + vector[i][1] * frame.y + vector[i][2] * frame.z
            converted_vectors.append(v_i)
        return converted_vectors


def vec_to_list(vector):
    if vector != 0:
        return [float(vector.args[0][0][i]) for i in range(0, 3)]
    else:
        return [0, 0, 0]


def generate_zero_vectors(n=1):
    if n > 0:
        vectors = [[0, 0, 0] for _ in range(n)]
        return vectors
    else:
        return [0, 0, 0]


def transform_vector(vector, from_frame, to_frame):
    DCM = to_frame.dcm(from_frame)
    transformed_vector = vectorize_list(np.matmul(DCM, np.asarray(vec_to_list(vector)).transpose()), to_frame)
    return transformed_vector


def cross_pdt(v1, v2):
    """

    :param v1: vector 1
    :param v2: vector 2
    :return: cross pdt
    """
    if v1 == 0 or v2 == 0:
        return 0
    else:
        frame = v1.args[0][1]
        v1, v2 = vec_to_list(v1), vec_to_list(v2)
        result = [v1[1] * v2[2] - v1[2] * v2[1],
                  v1[2] * v2[0] - v1[0] * v2[2],
                  v1[0] * v2[1] - v1[1] * v2[0]]
        return vectorize_list(result, frame)


def net_vel_update_for_component(component, blade=None, just_blade=False, just_hub=False):
    ref = component.rotary_ref_frames
    if just_hub is True:
        simplified_net_vel_code(ref[0])
        # update_net_vel(ref[0])

    if just_blade is True:
        if blade is not None:
            for i in range(len(ref[blade])):
                simplified_net_vel_code(ref[blade][i])
                # update_net_vel(ref[blade][i])

        else:
            blade = len(ref) - 1
            length = len(ref[blade])
            for j in range(1, blade + 1):
                for k in range(length):
                    simplified_net_vel_code(ref[j][k])
                    # update_net_vel(ref[j][k])

    return 0

#
# def update_net_vel(child_frame, acceleration=False):
#     if child_frame.name != Gb.body_fixed_cg.name:
#
#         parent_frame = child_frame.__dict__['pv_parent']
#         v_frame = child_frame.__dict__['v_naught']
#         ang_v_frame = child_frame.__dict__['w_frame']
#         parent_velocity = parent_frame.__dict__['v_net']
#         parent_ang_vel = parent_frame.__dict__['w_net']
#         pv_of_frame = child_frame.__dict__['position_vector']
#
#         v_frame += transform_vector(parent_velocity, parent_frame, child_frame) + transform_vector(cross_pdt(parent_ang_vel, pv_of_frame), parent_frame, child_frame)
#         ang_v_frame += transform_vector(parent_ang_vel, parent_frame, child_frame)
#         child_frame.__dict__.update({'v_net': v_frame, 'w_net': ang_v_frame})
#
#         if acceleration is True:
#             acc_frame = child_frame.__dict__['a_naught']
#             alpha_frame = child_frame.__dict__['alpha_frame']
#             parent_trans_acc = parent_frame.__dict__['a_net']
#             parent_ang_acc = parent_frame.__dict__['alpha_net']
#
#             if parent_ang_vel != 0 and pv_of_frame != 0:
#                 acc_frame += transform_vector(cross_pdt(parent_ang_vel, cross_pdt(parent_ang_vel, pv_of_frame)),
#                                               parent_frame, child_frame)
#                 if parent_ang_acc != 0:
#                     alpha_frame += transform_vector(parent_ang_acc, parent_frame, child_frame)
#                     acc_frame += transform_vector(cross_pdt(parent_ang_acc, pv_of_frame), parent_frame, child_frame)
#
#             if parent_trans_acc != 0:
#                 acc_frame += transform_vector(parent_trans_acc, parent_frame, child_frame)
#
#             child_frame.__dict__.update({'a_net': acc_frame, 'alpha_net': alpha_frame})
#
#     else:
#         child_frame.__dict__['a_net'] = child_frame.__dict__['a_naught'] + cross_pdt(child_frame.__dict__['w_frame'], child_frame.__dict__['v_naught'])
#         child_frame.__dict__['alpha_net'], child_frame.__dict__['w_net'] = child_frame.__dict__['alpha_frame'], child_frame.__dict__['w_frame']
#         return child_frame.__dict__['v_net'], child_frame.__dict__['a_net']
#
#     return v_frame, ang_v_frame


def update_net_vel(child_frame, acceleration=False):
    if child_frame.name != Gb.body_fixed_cg.name:
        parent_frame = child_frame.__dict__['pv_parent']
        v_frame = child_frame.__dict__['v_naught']
        ang_v_frame = child_frame.__dict__['w_frame']
        parent_velocity = parent_frame.__dict__['v_net']
        parent_ang_vel = parent_frame.__dict__['w_net']
        pv_of_frame = child_frame.__dict__['position_vector']

        if parent_velocity != 0:        # If parent has net trans velocity
            v_frame += transform_vector(parent_velocity, parent_frame, child_frame)

        if parent_ang_vel != 0:     # If parent has ang velocity
            v_frame += transform_vector(cross_pdt(parent_ang_vel, pv_of_frame), parent_frame, child_frame)
            ang_v_frame += transform_vector(parent_ang_vel, parent_frame, child_frame)
        child_frame.__dict__.update({'v_net': v_frame, 'w_net': ang_v_frame})

        if acceleration is True:
            acc_frame = child_frame.__dict__['a_naught']
            alpha_frame = child_frame.__dict__['alpha_frame']
            parent_trans_acc = parent_frame.__dict__['a_net']
            parent_ang_acc = parent_frame.__dict__['alpha_net']

            if parent_ang_vel != 0 and pv_of_frame != 0:
                acc_frame += transform_vector(cross_pdt(parent_ang_vel, cross_pdt(parent_ang_vel, pv_of_frame)), parent_frame, child_frame)
                if parent_ang_acc != 0:
                    alpha_frame += transform_vector(parent_ang_acc, parent_frame, child_frame)
                    acc_frame += transform_vector(cross_pdt(parent_ang_acc, pv_of_frame), parent_frame, child_frame)

            if parent_trans_acc != 0:
                acc_frame += transform_vector(parent_trans_acc, parent_frame, child_frame)

            child_frame.__dict__.update({'a_net': acc_frame, 'alpha_net': alpha_frame})

        return v_frame, ang_v_frame


def simplified_net_vel_code(frame):
    pv_parent = frame.__dict__['pv_parent']
    v_net = pv_parent.__dict__['v_net'] + cross_pdt(pv_parent.__dict__['w_net'], frame.__dict__['position_vector'])
    frame.__dict__['v_net'] = transform_vector(v_net, pv_parent, frame)
    frame.__dict__['w_net'] = frame.__dict__['w_frame'] + transform_vector(pv_parent.__dict__['w_net'], pv_parent, frame)
    return frame.__dict__['v_net']


def update_net_vel_for_all_stations(station_ref_frame_list, acceleration=False):
    for i in range(0, len(station_ref_frame_list)):
        update_net_vel(station_ref_frame_list[i], acceleration=acceleration)
    return 0


def find_loads(child_frame, parent_frame):
    pv_parent = child_frame.__dict__.get('pv_parent')
    count = 0

    while pv_parent.name != parent_frame.name:
        pv_parent = pv_parent.__dict__['pv_parent']
        if pv_parent.__dict__['position_vector'] != 0:
            break
        if count > 20:
            raise RuntimeError('Child - Parent relation not preserved')
        count += 1

    if pv_parent.name == parent_frame.name:
        forces, moments = child_frame.__dict__.get('frame_loads')
        forces, moments = vectorize_list([forces, moments], child_frame, 2)
        f_trans_parent = transform_vector(forces, child_frame, pv_parent)
        m_trans_parent = transform_vector(moments, child_frame, pv_parent)
        m = cross_pdt(child_frame.__dict__['position_vector'], f_trans_parent)
        f_trans_parent = vec_to_list(f_trans_parent)
        m_trans_parent = vec_to_list(m_trans_parent + m)
        return np.array((f_trans_parent, m_trans_parent))
    else:
        raise ValueError('Child - Parent relation not preserved')


def get_dcm(axis, angle):

    if axis == 1:
        return Matrix([[1, 0, 0],
                       [0, cos(angle), -sin(angle)],
                       [0, sin(angle), cos(angle)]])
    elif axis == 2:
        return Matrix([[cos(angle), 0, sin(angle)],
                       [0, 1, 0],
                       [-sin(angle), 0, cos(angle)]])
    elif axis == 3:
        return Matrix([[cos(angle), -sin(angle), 0],
                       [sin(angle), cos(angle), 0],
                       [0, 0, 1]])


def orient_with_axis(child_frame, parent_frame, axis, angle, angular_velocity=None):
    """

    :param child_frame:
    :param parent_frame:
    :param axis: string - "x", "y" or "z"
    :param angle: in radians
    :return: 0
    """
    axis = axis.lower()
    if axis == 'x':
        # F2.orient_axis(F1, F1.x, angle)
        DCM = get_dcm(1, angle)
    if axis == 'y':
        DCM = get_dcm(2, angle)
    if axis == 'z':
        DCM = get_dcm(3, angle)
    # DCM = F1.dcm(F2)
    child_frame.__dict__.get('_dcm_dict').update({parent_frame: DCM.transpose()})
    parent_frame.__dict__.get('_dcm_dict').update({child_frame: DCM})

    if angular_velocity is not None:
        child_frame.__dict__['w_frame'] = angular_velocity * child_frame.x
        pass
    return 0


def orient_azimuth_with_psi(parent_frame, child_frame, component, blade, psi):
    if psi == 0:
        psi_k = float(((2 * PI) * (blade - 1) / component.blade_no))
    else:
        psi_k = float(((2 * PI) * (blade - 1) / component.blade_no)) + psi

    if component.omega < 0:
        psi_k = -psi_k

    DCM = get_dcm(3, psi_k)

    'Updating dictionary with new pv'
    child_frame.__dict__.get('_dcm_dict').update({parent_frame: DCM.transpose()})
    parent_frame.__dict__.get('_dcm_dict').update({child_frame: DCM})
    return 0


def orient_euler(child_frame, parent_frame, euler):
    """

    :param child_frame: child reference frame
    :param euler: 3 angles in degrees in the order of ZYX
    :param parent_frame: Frame wrt which rotation is done
    :return: DCM of rotation
    """

    euler = [np.radians(euler[x]) for x in range(len(euler))]
    DCM = get_dcm(1, 0)
    for i in range(3):
        if euler[i] != 0:
            DCM *= get_dcm(3-i, euler[i])
    child_frame.__dict__.get('_dcm_dict').update({parent_frame: DCM.transpose()})
    parent_frame.__dict__.get('_dcm_dict').update({child_frame: DCM})

    return DCM


def set_7_vectors(vectors_list, parent_frame, child_frame):
    """

    :param vectors_list: Position vector, Translational Velocity and Acceleration
            Angular Velocity and Acceleration as a list of lists.
    :param parent_frame: Frame with respect to which all these vectors are defined
    :param child_frame: Frame whose properties these vectors define

    :return: 0
    """

    pos_vec = vectorize_list(vectors_list[0], parent_frame)
    forces, moments = vectors_list[-2], vectors_list[-1]
    del vectors_list[0], vectors_list[-2], vectors_list[-1]
    trans_vel, trans_acc, ang_vel, ang_acc = vectorize_list(vectors_list, child_frame, 4)
    vector_dict = {'position_vector': pos_vec, 'v_naught': trans_vel, 'a_naught': trans_acc, 'w_frame': ang_vel,
                   'alpha_frame': ang_acc, 'v_net': 0, 'w_net': 0,
                   'a_net': 0, 'alpha_net': 0, 'frame_loads': [forces, moments],
                   'frame_loads_net': 0, 'pv_parent': parent_frame}
    child_frame.__dict__.update(vector_dict)
    return 0


def create_cg_frame(v=None, a=None, omega=None, alpha=None):
    if v is None:
        v = zero_vector_list
    if a is None:
        a = zero_vector_list
    if omega is None:
        omega = zero_vector_list
    if alpha is None:
        alpha = zero_vector_list

    body_fixed_cg = vec.ReferenceFrame('body_fixed_cg')
    vectors_in_frame = [zero_vector_list, v, a, omega, alpha]
    pos_vec_cg_cg, trans_vel_cg_cg, trans_acc_cg_cg, ang_vel_cg_cg, ang_acc_cg_cg = vectorize_list(vectors_in_frame,
                                                                                                   body_fixed_cg, 5)

    velocity_dict = {'position_vector': pos_vec_cg_cg, 'v_naught': trans_vel_cg_cg, 'a_naught': trans_acc_cg_cg,
                     'w_frame': ang_vel_cg_cg,
                     'alpha_frame': ang_acc_cg_cg, 'v_net': trans_vel_cg_cg,
                     'a_net': trans_acc_cg_cg + ang_vel_cg_cg.cross(trans_vel_cg_cg),
                     'w_net': ang_vel_cg_cg, 'alpha_net': ang_acc_cg_cg, 'frame_loads': [generate_zero_vectors(2)],
                     'frame_loads_net': [generate_zero_vectors(2)]}

    body_fixed_cg.__dict__.update(velocity_dict)
    Gb.body_fixed_cg = body_fixed_cg

    return body_fixed_cg


def create_rotating_component(rotary_components):
    global shaft_angle, t, theta_i, beta_k, lag_k, torsion_k
    for rotary_component in rotary_components:
        component_name, euler, locations, blade_no, rpm, omega, radius, cd0, cm, sections, station_data = \
            itemgetter('component_name', 'euler', 'locations', 'blade_no', 'rpm', 'omega', 'radius', 'cd0', 'cm', 'stations', 'station_data')(rotary_component.__dict__)
        ref_frames_list, data_of_stations = [], []

        # Reference frame for the component itself
        hub_fixed_nr = vec.ReferenceFrame(component_name + '_hub_fixed')
        # orient_with_axis(hub_fixed_nr, Gb.body_fixed_cg, 'y', (sm.pi + shaft_angle))
        'Euler angles in degrees'
        orient_euler(hub_fixed_nr, Gb.body_fixed_cg, euler)
        vectors_in_this_frame = generate_zero_vectors(7)
        vectors_in_this_frame[0] = locations[0]
        set_7_vectors(vectors_in_this_frame, Gb.body_fixed_cg, hub_fixed_nr)
        rotary_component.rotary_ref_frames.append(hub_fixed_nr)
        ref_frames_list.append(hub_fixed_nr)

        hinge_offset = locations[1][0]  # Hinge offset only along the radial direction of the blade
        for i in range(1, len(locations) - 1):
            if locations[i + 1][0] > hinge_offset:
                hinge_offset = locations[i + 1][0]

        # Station ref frame data calculation once per component
        stations = np.linspace(hinge_offset + 0.05 * hinge_offset, radius, Gb.RotatingComponent.no_of_stations)
        for x in range(len(stations)):
            for k in range(len(sections) - 1):
                if sections[k][0] <= stations[x] <= sections[k + 1][0]:
                    chord = interpolate(sections[k][0], sections[k][1], sections[k + 1][0], sections[k + 1][1], stations[x])
                    sweep_angle = interpolate(sections[k][0], sections[k][2], sections[k + 1][0], sections[k + 1][2], stations[x])
                    theta = interpolate(sections[k][0], sections[k][3], sections[k + 1][0], sections[k + 1][3], stations[x])

                    rotary_component.station_data.append([stations[x], chord, sweep_angle, theta])
                    # data_of_stations.append([stations[x], theta])

        # Reference frames for the blades
        for i in range(1, blade_no + 1):
            frames_at_stations = []
            frames_for_blades = []

            # hub fixed rotating reference frame - azimuth
            hub_fixed_1k = vec.ReferenceFrame(component_name + '_hub_fixed_blade_' + str(i) + '_azimuth')
            psi_k = float(((2 * PI) * (i - 1) / blade_no))
            if omega < 0:
                psi_k = -psi_k
            orient_with_axis(hub_fixed_1k, hub_fixed_nr, 'z', psi_k)
            vectors_in_this_frame = generate_zero_vectors(7)
            vectors_in_this_frame[0] = zero_vector_list
            vectors_in_this_frame[3] = rpm
            ' Which direction is positive? '
            set_7_vectors(vectors_in_this_frame, hub_fixed_nr, hub_fixed_1k)
            ref_frames_list.append(hub_fixed_1k)
            frames_for_blades.append(hub_fixed_1k)

            """
            Defining the angles wrt ref_frames_list[-1] has nullified the point of chain definitions of ref frames 
            thus causing problems in dcm and every calculation made using that.
            """

            # blade fixed reference frames
            location_of_origin = [[0, 0, 0], locations[1], locations[2], locations[3]]
            angles_wrt_parent = [0, beta_k, lag_k, torsion_k]
            angle_text = ['_pitch', '_flap', '_lag', '_torsion']
            location_text = ['_theta_i', '_flap_hinge', '_lag_hinge', '_torsion_bearing']
            axis_of_rotation = ['x', 'y', 'z', 'x']
            'No need to define azimuth beyond blade fixed rotating RFs'
            # location_of_origin = [locations[1]]
            # angles_wrt_parent = [beta_k]
            # angle_text = ['_flap']
            # location_text = ['_flap_hinge']
            # axis_of_rotation = ['y']
            # if any(locations[2]) is True:
            #     location_of_origin.append(locations[2])
            #     angles_wrt_parent.append(lag_k)
            #     angle_text.append('_lag_hinge')
            #     location_text.append('_lag_hinge')
            #     axis_of_rotation.append('z')
            # if any(locations[3]) is True:
            #     location_of_origin.append(locations[3])
            #     angles_wrt_parent.append(torsion_k)
            #     angle_text.append('_pitch')
            #     location_text.append('_pitch_bearing')
            #     axis_of_rotation.append('x')

            for j in range(len(location_of_origin)):
                'hub fixed 1k --> flap --> lag --> torsion'
                frame = vec.ReferenceFrame(component_name + location_text[j] + '_blade_' + str(i) + angle_text[j])
                orient_with_axis(frame, ref_frames_list[-1], axis_of_rotation[j], angles_wrt_parent[j])
                vectors_in_this_frame = generate_zero_vectors(7)
                vectors_in_this_frame[0] = location_of_origin[j]
                if j > 0:
                    vectors_in_this_frame[0] = np.subtract(location_of_origin[j], location_of_origin[j-1])
                else:
                    vectors_in_this_frame[0] = location_of_origin[j]
                set_7_vectors(vectors_in_this_frame, ref_frames_list[-1], frame)
                ref_frames_list.append(frame)
                frames_for_blades.append(frame)

            # Creating reference frames at stations
            angle_text = ['_pitch']
            stations = np.linspace(hinge_offset + 0.05 * hinge_offset, radius, Gb.RotatingComponent.no_of_stations)
            for x in range(len(stations)):
                location_of_origin = [stations[x] - hinge_offset, 0, 0]
                for m in range(len(angle_text)):
                    frame = vec.ReferenceFrame(component_name + '_blade_' + str(i) + f'_station_{x + 1}' + angle_text[m])
                    # theta = rotary_component.station_data[x][3]
                    frame.__dict__['chord'] = rotary_component.station_data[x][1]
                    theta = 0
                    orient_with_axis(frame, ref_frames_list[-1], 'x', np.radians(theta))
                    vectors_in_this_frame = generate_zero_vectors(7)
                    vectors_in_this_frame[0] = location_of_origin
                    set_7_vectors(vectors_in_this_frame, ref_frames_list[-1], frame)
                    frames_at_stations.append(frame)

            rotary_component.rotary_ref_frames.append(frames_for_blades)
            rotary_component.stations_ref_frames.append(frames_at_stations)
        Gb.station_data.append([component_name, data_of_stations])
    return 0


if __name__ == '__main__':
    print('I am the ReferenceFrames.py file!\n')
    set_angles()
