import math
from time import time
import Global as Gb
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


error_plot = []


def single2var_gauss_quadrature(f, a=0, b=0, point_y=0):  # 5 point Gaussian quadrature
    """ Performs Gaussian integration
    :param f: Integrand function --> f(x,y)
    :param a: x lower limit
    :param b: x upper limit
    :param point_y: y value of the function
    :return: Gaussian 5 point single integral f(x,y)dx
    """
    transformation_factor_x = (b - a) * 0.5
    w = [0.568888888888889, 0.478628670499366, 0.478628670499366, 0.236926885056189, 0.236926885056189]
    x = [0, 0.538469310105683, -0.538469310105683, 0.906179845938664, -0.9061845938664]

    # Performing numerical integration
    value_x = 0
    for i in range(0, 5):
        point_x = 0.5 * ((x[i] * (b - a)) + b + a)
        value_x += w[i] * f(point_x, point_y)
    value_x = transformation_factor_x * value_x
    return value_x


def double_gauss_quadrature(f, a=0, b=0, c=0, d=0):  # 5 point Gaussian quadrature
    """ Performs Gaussian integration
    :param f: Integrand function --> f(x,y)
    :param a: x lower limit of integration
    :param b: x upper limit
    :param c: y lower limit
    :param d: y upper limit
    :return: Gaussian 5 point double integral f(x,y)dx dy
    """
    transformation_factor_x = (b - a) * 0.5
    transformation_factor_y = (d - c) * 0.5
    w = [0.568888888888889, 0.478628670499366, 0.478628670499366, 0.236926885056189, 0.236926885056189]
    x = [0, 0.538469310105683, -0.538469310105683, 0.906179845938664, -0.9061845938664]

    # Performing numerical integration
    value_y = 0
    for j in range(0, 5):
        value_x = 0
        point_y = 0.5 * ((x[j] * (d - c)) + d + c)
        for i in range(0, 5):
            point_x = 0.5 * ((x[i] * (b - a)) + b + a)
            value_x += w[i] * f(point_x, point_y)
        value_x = transformation_factor_x * value_x
        value_y += w[j] * value_x
    value_y = transformation_factor_y * value_y

    # returning appropriate results for single and double integrals respectively
    if c == 0 and d == 0:
        return round(value_x, 5)
    else:
        return round(value_y, 5)


def simplified_newton_raphson(function, y):
    iterations = 10 ** 5
    tolerance = 10 ** -5
    delta = 0.00125
    error = 1
    i = 0
    m = 1
    while error >= tolerance and i <= iterations:
        f = function(y)
        delta_y = delta * y
        fy = function(y + delta_y)
        f_dash = (fy - f) / delta_y
        y_new = y - m * (f / f_dash)
        error = abs(function(y_new))
        # print(error)
        y = y_new
        i += 1
    if i < iterations:
        return y_new

    else:
        print('Value did not converge')
        return -1


def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    y = m * (x - x2) + y2
    return y


def simpsons_rule(f_vals, low_lim, up_lim):
    n = len(f_vals)
    if n % 2 == 0:
        raise 'No. of function entries not in even intervals!'

    else:
        h = (up_lim - low_lim) / (3 * n)        # This is h / 3 not h
        value = f_vals[0] + f_vals[n - 1]
        for i in range(1, n-1):
            if i % 2 == 0:
                value += 2 * f_vals[i]
            if i % 2 != 0:
                value += 4 * f_vals[i]
        value = h * value

    if math.isnan(value) is True:
        value = 0
    if -10 ** -5 < value < 10 ** -5:
        value = 0
    return value


def runge_kutta_solver(function, conditions, initial_input, eval_pt, step_size, other_inputs=None):
    """

    :param function: Must be defined as f(conditions, other_input_1, other_input_2, etc.)
    :param conditions: Must be a ndarray
    :param initial_input: t_0 or Psi_0 condition
    :param eval_pt: Essentially a determiner of no of iterations
    :param step_size:
    :param other_inputs: List of all other inputs in order
    :return: final value of solution
    """
    if other_inputs is None:
        other_inputs = [None]
    iterations = round((eval_pt - initial_input) / step_size)
    half_step_size = 0.5 * step_size
    # time_list, solution_list = [initial_input], [conditions]

    for _ in range(1, iterations + 1):
        k1 = function(conditions, initial_input, *other_inputs)
        k2 = function(np.add(conditions, k1 * half_step_size), initial_input + half_step_size, *other_inputs)
        k3 = function(np.add(conditions, k2 * half_step_size), initial_input + half_step_size, *other_inputs)
        k4 = function(np.add(conditions, k3 * step_size), initial_input + step_size, *other_inputs)
        k4_final = np.add(k1, 2 * np.add(k2, k3))
        conditions = np.add(conditions, (step_size / 6) * np.add(k4_final, k4))
        initial_input += step_size
        # print(conditions, initial_input)
        # solution_list.append(conditions)
        # time_list.append(initial_input + step_size)
    return conditions
    # return conditions, solution_list, time_list


def newton_raphson_root_finder(function, y, others):
    iterations = 10 ** 3
    tolerance = 10 ** -5
    # damping = 1
    y_old = y.copy()
    y_new = y
    error = 1
    i = 0
    while np.amax(error) >= tolerance and i <= iterations:
        if len(y) > 1:
            start = time()
            f = function(y_old, *others, save_loads=True)
            print(f'Acceleration with initial guesses:\t{f}')
            Gb.trim_file.write('\n\nIteration No: ' + str(i+1) + '\n\nInitial Acceleration:\t' + str(f))
            Gb.section_file.write('\n\nIteration No: ' + str(i+1) + '\n\nInitial Acceleration:\t' + str(f))
            Gb.components_file.write('\n\nIteration No: ' + str(i+1))
        else:
            f = function(y_old, *others)

        if np.amax(f) <= tolerance:
            y_new = y_old
            break

        inv_jacob = inv_jacobian(function, y_old, f, others)
        mat = np.array(np.matmul(inv_jacob, np.transpose(np.matrix(f))))
        y_new = np.subtract(np.transpose(np.matrix(y_old)), mat)
        if is_iterable(f):
            if len(y_new) > 1:
                print(f'\nTime for this iteration:\t{(time()-start) / 60} min')
                y_new = list(map(float, y_new))
                Gb.trim_file.write(f'Time for this iteration:\t{(time()-start) / 60} min')
                Gb.components_file.write(f'Time for this iteration:\t{(time()-start) / 60} min')
                Gb.section_file.write(f'Time for this iteration:\t{(time()-start) / 60} min')

        error = np.absolute(function(y_new, *others))
        error_plot.append(error)
        y_old = y_new
        i += 1

    if i < iterations:
        if len(y_new) > 1:
            print('Acceleration with new guesses:\t', error)
            Gb.components_file.write('* * '* 50 + '\n')
            Gb.section_file.write('- * - * '* 50 + '\n')
            Gb.trim_file.write(f'\n\nFinal Inputs (radians): {str(y_new)}\nFinal Inputs (degrees): {str(list(map(np.degrees, y_new)))}\nAcceleration with new guesses:\t' + str(error))
        return y_new
    else:
        print('Value did not converge')
        return 1


def jacobian(function, y, fun_val_at_y, others):
    n = len(y)
    jacob = np.zeros((n, n))
    x = y.copy()
    for j in range(0, n):
        if x[j] == 0:
            delta_x = 0.00125
            x[j] += delta_x
        else:
            delta = 0.000125
            delta_x = delta * x[j]
            x[j] += delta_x
        if n == 1:
            fx = function(x, *others)

        if n > 1:
            if j < 3:
                print('\nTail rotor turned off')
                fx = function(x, *others, tail=False)
            elif j == 3:
                print('\nMain rotor turned off')
                fx = function(x, *others, main=False)
            else:
                print('Both on')
                fx = function(x, *others, tail=False)
            print(f'Acceleration with initial guesses:\t{fun_val_at_y}\nAcceleration after changing guess indexed {j}:\t{fx}')
            Gb.trim_file.write(f'\nAcceleration after changing guess indexed {str(j)}:\t{str(fx)}')
            if j == 5:
                print('*' * 60)

        df = (np.subtract(fx, fun_val_at_y)) / delta_x
        x[j] = y[j]
        if n > 1:
            for k in range(n):
                jacob[j][k] = df[k]
        else:
            jacob[j] = df
    return jacob.transpose()


def inv_jacobian(function_list, y, fun_val_at_y,  others):
    jacobian_matrix = jacobian(function_list, y, fun_val_at_y, others)
    inverse = inv(jacobian_matrix)

    if jacobian_matrix.shape[0] > 1:
        print('\nJacobian:')
        print(jacobian_matrix)
        print('\nInverse:')
        print(inverse)
        Gb.trim_file.write('\n\nJacobian Matrix:\n' + str(jacobian_matrix) + '\nInverse Jacobian Matrix:\n' + str(inverse) + '\n' + '* * ' * 50 + '\n')
    return inverse


# def newton_raphson_root_finder(function, y, others):
#     iterations = 10 ** 3
#     tolerance = 10 ** -5
#     # damping = 1
#     y_old = y.copy()
#     y_new = y
#     error = 1
#     i = 0
#     while np.amax(error) >= tolerance and i <= iterations:
#         if len(y) > 1:
#             start = time()
#             f = function(y_old, *others, save_loads=True)
#             print(f'Acceleration with initial guesses:\t{f}')
#             Gb.trim_file.write('\n\nIteration No: ' + str(i+1) + '\n\nInitial Acceleration:\t' + str(f))
#             Gb.components_file.write('\n\nIteration No: ' + str(i+1) + '\n\nInitial Acceleration:\t' + str(f))
#         else:
#             f = function(y_old, *others)
#
#         inv_jacob = inv_jacobian(function, y_old, f, others)
#         mat = np.array(np.matmul(inv_jacob, np.transpose(np.matrix(f))))
#         y_new = np.subtract(np.transpose(np.matrix(y_old)), mat)
#         if is_iterable(y_new):
#             if len(y_new) > 1:
#                 print(f'\nTime for this iteration:\t{(time()-start) / 60} min')
#                 y_new = list(map(float, y_new))
#                 Gb.trim_file.write(f'Time for this iteration:\t{(time()-start) / 60} min')
#
#         error = np.absolute(function(y_new, *others))
#         error_plot.append(error)
#         y_old = y_new
#         i += 1
#
#     if i < iterations:
#         if len(error) > 1:
#             print('Acceleration with new guesses:\t', error)
#             Gb.trim_file.write(f'\n\nFinal Inputs (radians): {str(y_new)}\nFinal Inputs (degrees): {str(list(map(np.degrees, y_new)))}\nAcceleration with new guesses:\t' + str(error))
#         return y_new
#     else:
#         print('Value did not converge')
#         return 1
#
#
# def jacobian(function, y, fun_val_at_y, others):
#     n = len(y)
#     jacob = np.zeros((n, n))
#     x = y.copy()
#     for j in range(0, n):
#         if x[j] == 0:
#             delta_x = 0.00125
#             x[j] += delta_x
#         else:
#             delta = 0.000125
#             delta_x = delta * x[j]
#             x[j] += delta_x
#         if n == 1:
#             fx = function(x, *others)
#
#         if n > 1:
#             if j < 3:
#                 print('\nTail rotor turned off')
#                 fx = function(x, *others, tail=False)
#             elif j == 3:
#                 print('\nMain rotor turned off')
#                 fx = function(x, *others, main=False)
#             else:
#                 print('Both on')
#                 fx = function(x, *others, tail=False)
#             print(f'Acceleration with initial guesses:\t{fun_val_at_y}\nAcceleration after changing guess indexed {j}:\t{fx}')
#             Gb.trim_file.write(f'\nAcceleration after changing guess indexed {str(j)}:\t{str(fx)}')
#             if j == 5:
#                 print('*' * 60)
#
#         df = (np.subtract(fx, fun_val_at_y)) / delta_x
#         x[j] = y[j]
#         if n > 1:
#             for k in range(n):
#                 jacob[j][k] = df[k]
#         else:
#             jacob[j] = df
#     return jacob.transpose()
#
#
# def inv_jacobian(function_list, y, fun_val_at_y,  others):
#     jacobian_matrix = jacobian(function_list, y, fun_val_at_y, others)
#     inverse = inv(jacobian_matrix)
#
#     if jacobian_matrix.shape[0] > 1:
#         print('\nJacobian:')
#         print(jacobian_matrix)
#         print('\nInverse:')
#         print(inverse)
#         Gb.trim_file.write('\n\nJacobian Matrix:\n' + str(jacobian_matrix) + '\nInverse Jacobian Matrix:\n' + str(inverse) + '\n' + '* * ' * 50 + '\n')
#     return inverse


def plotter():
    x = [i for i in range(1, len(error_plot) + 1)]
    plt.plot(x, error_plot)
    plt.show()
    return 0


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

