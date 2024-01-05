
import numpy as np
import matplotlib.pyplot as plt
from time import time

import Atmosphere as Atm
import Global as Gb
import GetInput as Gip
import ReferenceFrames as Ref
import NumericalMethods as Num
import Aerodynamics as Flap
import RotorModel as Rot


def estimate_aerodynamic_loads(component, omega, psi, cyclic_input, inflow=None, inflow_type=1, write=False):
    if inflow is None:
        inflow = [0, 0, 0]

    def integrate_loads():
        """

        :return: [ [drag, side force, lift], [mx, my, mz] ]
        """
        total_x = Num.simpsons_rule(x_force_at_hinge, component.station_data[0][0], component.station_data[-1][0])
        total_y = Num.simpsons_rule(y_force_at_hinge, component.station_data[0][0], component.station_data[-1][0])
        total_z = Num.simpsons_rule(z_force_at_hinge, component.station_data[0][0], component.station_data[-1][0])
        total_moment_x = Num.simpsons_rule(x_moment_at_hinge, component.station_data[0][0], component.station_data[-1][0])
        total_moment_y = Num.simpsons_rule(y_moment_at_hinge, component.station_data[0][0], component.station_data[-1][0])
        total_moment_z = Num.simpsons_rule(z_moment_at_hinge, component.station_data[0][0], component.station_data[-1][0])
        total_forces = [total_x, total_y, total_z]
        total_moments = [total_moment_x, total_moment_y, total_moment_z]
        return [total_forces, total_moments]

    def flap_calculation(component, blade_no, psi, d_psi, aerodynamic_loads, beta_guess=None):
        if beta_guess is None:
            beta_guess = [-np.radians(3), 0]
        i = blade_no
        beta = component.rotary_ref_frames[i + 1][1].__dict__.get('flap_angle', beta_guess[0])
        beta_dot = Ref.vec_to_list(component.rotary_ref_frames[i + 1][1].__dict__['w_frame'])[1]
        print(f"blade: {i + 1}\ninit_beta: {np.degrees(float(beta))}\ninit_beta_dot: {np.degrees(float(beta_dot))}")
        rkstart = time()
        new_beta, new_beta_dot = Num.runge_kutta_solver(Flap.beta_double_dot, [beta, beta_dot], psi / component.omega,
                                                        (psi + d_psi) / component.omega, d_psi / component.omega,
                                                        [component, i, aerodynamic_loads[1][1],
                                                         component.rotary_ref_frames[i + 1][1],
                                                         component.rotary_ref_frames[i + 1][-1]])
        print('Time for rk4: ', time() - rkstart)
        component.rotary_ref_frames[i + 1][1].__dict__['flap_angle'] = new_beta
        component.rotary_ref_frames[i + 1][1].__dict__['w_frame'] = new_beta_dot * component.rotary_ref_frames[i + 1][1].y
        component.rotary_ref_frames[i + 1][1].__dict__['frame_loads'] = aerodynamic_loads
        beta_dob_dot = Flap.beta_double_dot([new_beta, new_beta_dot], psi, component, i, aerodynamic_loads[1][1], component.rotary_ref_frames[i + 1][1], component.rotary_ref_frames[i + 1][-1])
        print(f"beta: {np.degrees(float(new_beta))}\nbeta_dot: {np.degrees(float(beta_dob_dot[0]))}\nbeta_2dot: {np.degrees(float(beta_dob_dot[1]))}\n")
        return new_beta, *beta_dob_dot

    ref = component.stations_ref_frames

    for i in range(len(ref)):   # One iteration per blade
        station_index = 0
        x_force_at_hinge, y_force_at_hinge, z_force_at_hinge, x_moment_at_hinge, y_moment_at_hinge, z_moment_at_hinge = [], [], [], [], [], []
        hinge_offset = component.locations[-1][0]
        Gb.section_file.write('\n\nblade ' + str(i + 1) + '\n')
        # Gb.components_file.write('\n\nblade ' + str(i + 1) + '\n')

        if isinstance(cyclic_input, list):
            psi_k = float((2 * Gb.PI * i / component.blade_no) + psi)
            cyclic_theta = cyclic_input[0] + (cyclic_input[1] * np.cos(psi_k)) + (cyclic_input[2] * np.sin(psi_k))
            angular_vel = (-cyclic_input[1] * np.sin(psi_k)) + (cyclic_input[2] * np.cos(psi_k)) * component.omega
            Ref.orient_with_axis(component.rotary_ref_frames[i + 1][1], component.rotary_ref_frames[i + 1][0], 'x', cyclic_theta, angular_velocity=angular_vel)
        else:
            cyclic_theta = cyclic_input
            psi_k = float((2 * Gb.PI * i / component.blade_no) + psi)
            Ref.orient_with_axis(component.rotary_ref_frames[i+1][1], component.rotary_ref_frames[i+1][0], 'x', cyclic_theta)
        Ref.net_vel_update_for_component(component, just_blade=True, blade=i+1)

        for j in range(len(ref[i])):  # Fetching the station pitch reference frames
            'Velocities at the fetched reference frames'
            frame = ref[i][j]
            vel = Ref.update_net_vel(frame, acceleration=False)[0]     # Changed from find net vel of to UPDATE NET VEL as it was redundant
            # vel = Ref.simplified_net_vel_code(frame)     # Changed from find net vel of to UPDATE NET VEL as it was redundant
            r_bar = (Ref.vec_to_list(frame.__dict__['position_vector'])[0] + hinge_offset) / component.radius
            local_inflow = Rot.final_inflow_value(inflow, Ref.vec_to_list(component.rotary_ref_frames[0].__dict__['v_net'])[-1], r_bar, psi_k, inflow_type=inflow_type, write=write)
            inflow_vector = local_inflow * omega * component.radius * component.rotary_ref_frames[0].z
            inflow_vector = Ref.transform_vector(inflow_vector, component.rotary_ref_frames[0], frame)
            vel = vel + inflow_vector
            theta = float(np.radians(component.station_data[station_index][-1]))
            Gb.section_file.write(str(j + 1) + '. section: ' + str(Ref.vec_to_list(frame.__dict__['position_vector'])[0]) + '\n')
            Gb.section_file.write('inflow ratio: ' + str(local_inflow) + '\tinflow velocity: ' + str(Ref.vec_to_list(inflow_vector)) + '\ttheta_I: ' + str(cyclic_theta))
            Flap.aerodynamic_load_at_station(component, frame, vel, theta)
            [x_force, y_force, z_force], [moment_x, moment_y, moment_z] = Ref.find_loads(frame, component.rotary_ref_frames[i+1][-1])
            # Gb.section_file.write(f"\t{str((x_force, y_force, z_force))}, {str((moment_x, moment_y, moment_z))}\n\n")
            x_force_at_hinge.append(x_force)
            y_force_at_hinge.append(y_force)
            z_force_at_hinge.append(z_force)
            x_moment_at_hinge.append(moment_x)
            y_moment_at_hinge.append(moment_y)
            z_moment_at_hinge.append(moment_z)
            # Loop for station_index
            station_index += 1
        v = integrate_loads()
        component.rotary_ref_frames[i + 1][-1].__dict__['frame_loads'] = v
        if component.rotary_ref_frames[i + 1][-1].name != component.rotary_ref_frames[i+1][2].name:
            component.rotary_ref_frames[i+1][2].__dict__['frame_loads'] = Ref.find_loads(component.rotary_ref_frames[i + 1][-1], component.rotary_ref_frames[i+1][2])
        # else:
        #     aerodynamic_loads = v
        # new_beta, new_beta_dot, beta_dob_dot = flap_calculation(component, i, psi, Gb.RotatingComponent.d_psi, aerodynamic_loads)
    # print(component.rotary_ref_frames[i+1][2].__dict__['frame_loads'])
    return component.rotary_ref_frames[i+1][2].__dict__['frame_loads']


def find_loads_at_cg(cyclic_input, others, main=True, tail=True, save_loads=False):
    beta = []
    dummy_b, dummy_bdot, dummy_b2dot = [], [], []
    b, b_dot, b_2dot = [], [], []
    lamda_array_inner = [], []
    lamda_array_outer = [], []
    inflow_convergence = [False, False]
    switch = [main, tail]
    if main is False:
        inflow_convergence[0] = True
    if tail is False:
        inflow_convergence[1] = True
    error_lamda = []
    inflow = [[], []]
    inflow_type = [2, 1]
    rho = Gb.Atmos.density
    d_psi = Gb.RotatingComponent.d_psi
    n = len(Gb.rotary_components)
    az = []

    psi = 0
    psi_loop_index = 0
    resultant_force = Ref.generate_zero_vectors(n)
    resultant_moment = Ref.generate_zero_vectors(n)
    load_coefficients = Ref.generate_zero_vectors(n)
    T = Gb.vehicle_parameters[0].mass * Gb.g
    start = time()

    for i in range(n):
        Ref.net_vel_update_for_component(Gb.rotary_components[i], just_hub=True)

    while psi <= (15 * 2 + 0.01) * Gb.PI:
        psi_deg = np.degrees(psi)
        # print(f'Psi = {psi_deg}')
        Gb.section_file.write('* * * ' * 50 + '\npsi = ' + str(psi_deg) + '\n')
        # Gb.components_file.write('* * * ' * 50 + '\npsi = ' + str(psi_deg))
        i = 0

        if switch[0] is False:
            i = 1

        while i < n:
            component = Gb.rotary_components[i]
            Gb.section_file.write(f'\ncomponent name:\t{component.component_name}')
            for blade in range(1, len(component.rotary_ref_frames)):
                Ref.orient_azimuth_with_psi(component.rotary_ref_frames[0], component.rotary_ref_frames[blade][0], component, blade, psi)

            if inflow_convergence[i] is False and switch[i] is True:
                if psi == 0:
                    if i == 0:
                        c_t = T / (rho * (component.radius ** 4) * Gb.PI * (component.omega ** 2))
                        load_coefficients[i] = [c_t, 0, 0]
                        initial_inflow = [np.sqrt(load_coefficients[i][0] / 2), 0, 0]     # 0.0605
                        # initial_inflow = [0.0605, 0, 0]
                    if i == 1:
                        load_coefficients[i] = [0.01, 0, 0]
                        initial_inflow = [0.0824, 0, 0]
                else:
                    initial_inflow = inflow[i]

                if switch[i] is True:
                    inflow[i] = Rot.inflow_calculation(component, load_coefficients[i], psi, inflow_type=inflow_type[i], initial_inflow=initial_inflow)
                    # Gb.inflow_file.write(f'{str(psi_deg)}, {str(inflow[0][0])}, {str(inflow[0][1])}, {str(inflow[0][2])}\n')

                # if i == 1 and switch[i] is True:
                #     inflow[i] = Rot.inflow_calculation(component, load_coefficients[i], psi, inflow_type=inflow_type[i], initial_inflow=initial_inflow)

                lamda_array_inner[i].append(inflow[i])

            if switch[i] is True:
                if i == 0:
                    estimate_aerodynamic_loads(component, component.omega, psi, cyclic_input[0:3], inflow=inflow[i], inflow_type=inflow_type[i], write=True)
                if i == 1:
                    estimate_aerodynamic_loads(component, component.omega, psi, cyclic_input[3], inflow=inflow[i], inflow_type=inflow_type[i])

            'Transfering the forces and moments to hub fixed nr frame'
            component_wise_loads = []
            for blade in range(component.blade_no):
                hub_loads = Ref.find_loads(component.rotary_ref_frames[blade + 1][2], component.rotary_ref_frames[0])
                Gb.section_file.write(f"\n{blade+1}:\t{str(hub_loads[0])}, {str(hub_loads[1])}\n")
                component_wise_loads.append(hub_loads)

            if inflow_convergence[i] is False and inflow_type[i] == 2 and switch[i] is True:
                f = [component_wise_loads[k][0][-1] for k in range(len(component_wise_loads))]
                f = np.sum(f)
                c_t = f / (rho * (Gb.rotary_components[i].radius ** 4) * Gb.PI * (Gb.rotary_components[i].omega ** 2))
                load_coefficients[i] = [float(c_t), 0, 0]
                if i == 0:
                    Gb.inflow_file.write(f'{str(psi_deg)}, {str(c_t)}, {str(f)}, {str(initial_inflow)}, {str(inflow[i])}\n')

            for x in range(len(component_wise_loads)):
                resultant_force[i] = np.add(resultant_force[i], component_wise_loads[x][0])
                resultant_moment[i] = np.add(resultant_moment[i], component_wise_loads[x][1])
            i += 1
            if switch[1] is False:
                i = 3

        az.append(psi * 180 / np.pi)

        if round(psi_deg + np.degrees(d_psi), 0) % 360 == 0 and psi != 0:
            for i in range(n):
                lamda_array_outer[i].append(lamda_array_inner[i])
                # print(len(lamda_array_outer[i]))
                if psi_deg > 360 and inflow_convergence[i] is False and switch[i] is True:
                    lamda_error = np.subtract(lamda_array_outer[i][-1], lamda_array_outer[i][-2])
                    difference_1 = 0
                    difference_2 = 0
                    difference_3 = 0
                    for j in range(len(lamda_error)):
                        difference_1 += (lamda_error[j][0]) ** 2
                        difference_2 += (lamda_error[j][1]) ** 2
                        difference_3 += (lamda_error[j][2]) ** 2
                    norm_error_lamda = [np.sqrt(difference_1), np.sqrt(difference_2), np.sqrt(difference_3)]
                    # print(f"Rotation no: {round(psi / (2 * Gb.PI), 0)}\tLambda Error Norm:\t{norm_error_lamda}")
                    if norm_error_lamda[0] < 10**-5 and norm_error_lamda[1] < 10**-5  and norm_error_lamda[2] < 10**-5 :
                        error_lamda.append(norm_error_lamda)
                        inflow_convergence[i] = True

            lamda_array_inner = [], []

        if round(psi_deg, 0) % 360 == 0 and psi != 0:
            print('psi:\t', psi_deg)
            for i in range(0, n):
                if switch[i] is True:
                    resultant_f = [h / (2 * Gb.PI / d_psi) for h in resultant_force[i]]
                    resultant_m = [h / (2 * Gb.PI / d_psi) for h in resultant_moment[i]]
                    Gb.rotary_components[i].rotary_ref_frames[0].__dict__['frame_loads'] = [resultant_f, resultant_m]
                # print(f"Loads after averaging for component {Gb.rotary_components[i].component_name}: {Gb.rotary_components[i].rotary_ref_frames[0].__dict__['frame_loads']}")

                if inflow_convergence[i] is False and inflow_type[i] == 3 and switch[i] is True:
                    c_t = resultant_f[2] / (rho * (Gb.rotary_components[i].radius ** 4) * Gb.PI * (Gb.rotary_components[i].omega ** 2))
                    c_mx = resultant_m[0] / (rho * (Gb.rotary_components[i].radius ** 5) * Gb.PI * (Gb.rotary_components[i].omega ** 2))
                    c_my = resultant_m[1] / (rho * (Gb.rotary_components[i].radius ** 5) * Gb.PI * (Gb.rotary_components[i].omega ** 2))
                    load_coefficients[i] = [float(c_t), round(c_mx, 5), round(c_my, 5)]

            # print(f'\nTime for calling one azimuthal rotation with psi = {psi_deg}:\t{(time() - azimuth_time) / 60} minutes')
            # azimuth_time = time()
            if all(inflow_convergence):
                print(f'\nTime for inflow convergence:\t{(time() - start) / 60} minutes')
                print(f"Rotation no: {round(psi / (2 * Gb.PI), 0)}\tLambda Error Norms:\t{error_lamda}")

                cg_loads = Ref.generate_zero_vectors(2)
                for i in range(n):
                    print(f"Loads after averaging for component {Gb.rotary_components[i].component_name}: {Gb.rotary_components[i].rotary_ref_frames[0].__dict__['frame_loads']}")
                    if i == 0:
                        if main is True:
                            cg_loads = np.add(Ref.find_loads(Gb.rotary_components[0].rotary_ref_frames[0], Gb.body_fixed_cg), cg_loads)
                        if save_loads is True:
                            Gb.rotary_components[0].rotary_ref_frames[0].__dict__['saved_loads'] = cg_loads
                        if main is False:
                            cg_loads = np.add(Gb.rotary_components[0].rotary_ref_frames[0].__dict__['saved_loads'], cg_loads)
                    if i == 1:
                        cg_loads = np.add(Ref.find_loads(Gb.rotary_components[1].rotary_ref_frames[0], Gb.body_fixed_cg), cg_loads)
                    Gb.components_file.write(f'\nInflow Converged {str(Gb.rotary_components[i].component_name)} loads in Hub Frame:\t' + str(Gb.rotary_components[i].rotary_ref_frames[0].__dict__['frame_loads']) + '\n')
                Gb.components_file.write('\nInflow Converged Component Loads in CG Frame:\t[' + str(cg_loads[0]) + f' {str(cg_loads[1])}]' + '\n')
                Gb.section_file.write('* * ' * 50 + '\n')
                break
            resultant_force, resultant_moment = Ref.generate_zero_vectors(n), Ref.generate_zero_vectors(n)
            # break

        psi = psi + d_psi
        psi_loop_index += 1

    # cg_loads = Ref.generate_zero_vectors(2)
    # for i in range(0, n):
    #     load = Ref.find_loads(Gb.rotary_components[i].rotary_ref_frames[0], Gb.body_fixed_cg)
    #     cg_loads = np.add(load, cg_loads)
    #     print()
    #     print(f'{Gb.rotary_components[i].component_name}:\t{load}')
    return cg_loads


if __name__ == '__main__':
    # print('I am MultiRotorVehicle.py\n')
    Atm.atm_cal()
    Gip.read_inputs(r'InputData.xlsx')
    v_cg = [10.8504, 0, 0]
    Ref.set_angles([0, 0, 0, 0, 0, 0])
    Ref.create_cg_frame(v_cg)
    Ref.create_rotating_component(Gb.rotary_components)
    start = time()
    # cyclic_input = [0.0174532925 * 8.21645, 0.0174532925 * 0.827019 * 1, 0.0174532925 * -0.255768 * 1, 0.0174532925 * 9.92829 * 1]
    cyclic_input = [0.12012134653005913, 0.027368535143513867, -0.014474601873315902, 0.1615008888957343, -0.003512541946376015, -0.02914150204392799]
    loads = find_loads_at_cg(cyclic_input, others=[None], main=True, tail=True)
    print('\nCG Loads: ', loads)
    # print(loads)
    print(f'\nTotal time:\t{(time() - start)} seconds')

