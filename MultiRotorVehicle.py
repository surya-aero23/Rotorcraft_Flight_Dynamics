# 23 - 03 - 23


import numpy as np
import matplotlib.pyplot as plt
import time as t

import Atmosphere as Atm
import Global as Gb
import GetInput as Gip
import ReferenceFrames as Ref
import NumericalMethods as Num
import Aerodynamics as Flap
import SectionalLoads as Sec
import RotorModel as Rot

lam0, lam1s, lam1c = [], [], []
beta = []
beta_blade = []
b, b_dot, b_2dot = [], [], []
eb, eb_dot, eb_2dot = [], [], []


def find_aerodynamic_loads(component, load_coefficients, omega, psi, v_cg, include_inflow=True):

    def integrate_loads():
        """

        :return: [ [drag, side force, lift], [mx, my, mz] ]
        """
        total_lift = Num.simpsons_rule(sectional_lift, component.station_data[0][0], component.station_data[-1][0])
        total_drag = Num.simpsons_rule(sectional_drag, component.station_data[0][0], component.station_data[-1][0])
        total_side_force = Num.simpsons_rule(sectional_side_force, component.station_data[0][0], component.station_data[-1][0])
        total_moment_x = Num.simpsons_rule(sectional_moment_x, component.station_data[0][0], component.station_data[-1][0])
        total_moment_y = Num.simpsons_rule(sectional_moment_y, component.station_data[0][0], component.station_data[-1][0])
        total_moment_z = Num.simpsons_rule(sectional_moment_z, component.station_data[0][0], component.station_data[-1][0])
        total_forces = [total_drag, total_side_force, total_lift]
        total_moments = [total_moment_x, total_moment_y, total_moment_z]
        return [total_forces, total_moments]

    inflow = 0
    ref = component.stations_ref_frames
    Ref.net_vel_update_for_component(component)

    for i in range(len(ref)):   # One iteration per blade
        station_index = 0
        sectional_lift, sectional_drag, sectional_side_force, sectional_moment_x, sectional_moment_y, sectional_moment_z = [], [], [], [], [], []
        cyclic_theta = Gb.theta_0 + Gb.theta_1c * np.cos(psi) + Gb.theta_1s * np.sin(psi)

        for j in range(len(ref[i])):  # Fetching the station pitch reference frames
            'Velocities at the fetched reference frames'
            frame = ref[i][j]
            vel = Ref.find_net_vel_of(component, frame, acceleration=False)
            'Adding inflow to u_p'
            if include_inflow is True:
                pv = frame.__dict__['position_vector']
                pv = pv.dot(frame.x)
                inflow = Rot.inflow_calculation(component, load_coefficients, v_hub=v_cg, psi=psi, c=3, r=pv, k=i + 1)
                inflow2 = inflow[1:]
                inflow = inflow[0] * omega * component.radius * component.rotary_ref_frames[0].z

            inflow = Ref.transform_vector(inflow, component.rotary_ref_frames[0], frame)
            vel = vel + inflow

            theta = float(np.radians(component.station_data[station_index][-1]) + cyclic_theta)
            Flap.aerodynamic_load_at_station(component, frame, vel, theta)
            [drag, side, lift], [moment_x, moment_y, moment_z] = Ref.find_loads(frame, component.rotary_ref_frames[i+1][-1])
            sectional_lift.append(lift)
            sectional_drag.append(drag)
            sectional_side_force.append(side)
            sectional_moment_x.append(moment_x)
            sectional_moment_y.append(moment_y)
            sectional_moment_z.append(moment_z)

            # Loop for station_index
            station_index += 1
            # print()
        v = integrate_loads()
        'Update the integrated loads to the torsion hinge frame to calculate flap hinge loads'
        component.rotary_ref_frames[i + 1][-1].__dict__['frame_loads'] = v
        aerodynamic_loads = Ref.find_loads(component.rotary_ref_frames[i + 1][-1], component.rotary_ref_frames[i+1][1])
        beta = component.rotary_ref_frames[i + 1][1].__dict__.get('flap_angle', -np.radians(3))
        beta_dot = Ref.vec_to_list(component.rotary_ref_frames[i + 1][1].__dict__['w_frame'])[1]
        print(f"blade: {i + 1}\ninit_beta: {np.degrees(float(beta))}\ninit_beta_dot: {np.degrees(float(beta_dot))}")
        new_beta, new_beta_dot = Num.runge_kutta_solver(Flap.beta_double_dot, [beta, beta_dot], psi / component.omega, (psi + d_psi) / component.omega, d_psi / component.omega,
                                                        [component, i, aerodynamic_loads[1][1], component.rotary_ref_frames[i + 1][1], component.rotary_ref_frames[i + 1][-1]])
        component.rotary_ref_frames[i + 1][1].__dict__['flap_angle'] = new_beta
        component.rotary_ref_frames[i + 1][1].__dict__['w_frame'] = new_beta_dot * component.rotary_ref_frames[i + 1][1].y
        component.rotary_ref_frames[i + 1][1].__dict__['frame_loads'] = aerodynamic_loads
        print(f"beta: {np.degrees(float(new_beta))}\nbeta_dot: {np.degrees(float(new_beta_dot))}\n")
        Ref.orient_with_axis(component.rotary_ref_frames[i + 1][1], component.rotary_ref_frames[i + 1][0], 'y', new_beta)
    beta_dob_dot = Flap.beta_double_dot([new_beta, new_beta_dot], psi, component, i, aerodynamic_loads[1][1], component.rotary_ref_frames[i + 1][1], component.rotary_ref_frames[i + 1][-1])
    lam0.append(inflow2[0])
    lam1s.append(inflow2[1])
    lam1c.append(inflow2[2])
    return new_beta, new_beta_dot, beta_dob_dot[-1]


if __name__ == '__main__':
    print('I am MultiRotorVehicle.py\n')

    Atm.atm_cal()
    Gip.read_inputs(r'InputData.xlsx')
    v_cg = [10, 0, 0]
    Ref.set_angles([0, 0, 0, 0, 0, 0])
    Ref.create_cg_frame(v_cg)
    Ref.create_rotating_component(Gb.rotary_components)

    rho = Gb.Atmos.density
    d_psi = Gb.RotatingComponent.d_psi
    n = len(Gb.rotary_components)
    blade_lift = []
    T = Gb.vehicle_parameters[0].mass * Gb.g / n
    start = t.time()

    comp_loads = []
    az = []
    overall_thrust, overall_mz = [], []
    load_coefficients = [None] * n
    count = 0
    inner_count = 0
    psi = 0

    while psi <= (8 * 2 + 0.01) * Gb.PI:
        print(f"\n\nPsi = {np.degrees(psi)} deg")
        thrust, mz = [], []
        for i in range(n):
            component = Gb.rotary_components[i]
            print(f"\nComponent name: {component.component_name}")
            a = t.time()
            resultant_force = [0, 0, 0]
            resultant_moment = [0, 0, 0]

            for blade in range(1, len(component.rotary_ref_frames)):
                for j in range(0, len(component.rotary_ref_frames[blade]), 4):
                    Ref.orient_azimuth_with_psi(component.rotary_ref_frames[0], component.rotary_ref_frames[blade][j], component, blade, psi)

            if psi == 0:
                c_t = T / (rho * (component.radius ** 4) * Gb.PI * (component.omega ** 2))
                c_mx, c_my = 0, 0
                load_coefficients[i] = [c_t, c_mx, c_my]
            beta.append(find_aerodynamic_loads(component, load_coefficients[i], component.omega, psi, v_cg))

            'Transfering the forces and moments to hub fixed nr frame'
            component_wise_loads = []
            for blade in range(component.blade_no):
                ab = Ref.find_loads(component.rotary_ref_frames[blade + 1][1], component.rotary_ref_frames[0])
                component_wise_loads.append(ab)

            for x in range(len(component_wise_loads)):
                resultant_force = np.add(resultant_force, component_wise_loads[x][0])
                resultant_moment = np.add(resultant_moment, component_wise_loads[x][1])

            c_t = resultant_force[2] / (rho * (component.radius ** 4) * Gb.PI * (component.omega ** 2))
            c_mx = resultant_moment[0] / (rho * (component.radius ** 5) * Gb.PI * (component.omega ** 2))
            c_my = resultant_moment[1] / (rho * (component.radius ** 5) * Gb.PI * (component.omega ** 2))
            load_coefficients[i] = [float(c_t), round(c_mx, 5), round(c_my, 5)]
            print(f"Forces: {resultant_force}\nMoments: {resultant_moment}\nTime for Component: {t.time() - a} seconds")

        az.append(psi * 180 / np.pi)

        # if round((psi / d_psi), 0) % 72 == 0:
        #     b.append([(beta[i][0]) for i in range(count, count + 71)])
        #     b_dot.append([(beta[i][1]) for i in range(count, count + 71)])
        #     b_2dot.append([(beta[i][2]) for i in range(count, count + 71)])
        #     count += 72
        #     if count > 72 * 2:
        #         inner_count += 1
        #         eb.append(np.subtract(b[inner_count - 1], b[inner_count]))
        #         eb_dot.append(np.subtract(b_dot[inner_count - 1], b_dot[inner_count]))
        #         eb_2dot.append(np.subtract(b_2dot[inner_count - 1], b_2dot[inner_count]))
        #         print(max(eb), max(eb_dot), max(eb_2dot), sep='\t')

        psi = psi + d_psi
        # break
    print(f"\nTime taken: {t.time() - start} sec")


beta_blade = [(180 * beta[i][0] / Gb.PI) for i in range(len(beta))]
beta_dot_blade = [(180 * beta[i][1] / Gb.PI) for i in range(len(beta))]
beta_dob_dot = [(180 * beta[i][2] / Gb.PI) for i in range(len(beta))]

plt.plot(az, beta_blade, label='beta')
plt.grid()
plt.legend()
plt.show()
plt.plot(az, beta_dot_blade, label='beta_dot')
plt.legend()
plt.grid()
plt.show()
plt.plot(az, beta_dob_dot, label='beta_double_dot')
plt.legend()
plt.grid()
plt.show()

# plt.plot(az, lam0, label='lam0')
# plt.plot(az, lam1s, label='lam1s')
# plt.plot(az, lam1c, label='lam1c')
# plt.legend()
# plt.show()
