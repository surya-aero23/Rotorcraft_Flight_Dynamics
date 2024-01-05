import numpy as np

import RotorModel as Rmod
from numpy.linalg import inv


sig = 0.1
a = 2 * np.pi
cd0 = 0.0079
C_w = 0.0032
lamda = 0.04
lock = 12
w_rf = 1.1
beta0 = 0.046145969
beta1c = 0
beta1s = 0


def thrust_coefficient(x):
    if mu == 0:
        return C_w
    else:
        theta0 = x[0]
        theta1c = x[1]
        theta1s = x[2]
        a0 = (sig * a / 6) * (1 + 1.5 * (mu ** 2))
        b0 = sig * a * mu * 0.25
        c0 = 0
        d0 = - sig * a * lamda * 0.25
        C_t = a0 * theta0 + b0 * theta1s + c0 * theta1c + d0
        return C_t


Rmod.mu = 0
mu = Rmod.mu
c_t = thrust_coefficient([0, 0, 0, 0])
Rmod.c_t = c_t
lamda = Rmod.inflow_calculation()


def f0(x):
    theta0 = x[0]
    theta1c = x[1]
    theta1s = x[2]
    alpha = x[3]
    a0 = (sig * a / 6) * (1 + 1.5 * (mu ** 2))
    b0 = sig * a * mu * 0.25
    c0 = 0
    d0 = - sig * a * lamda * 0.25
    c_t = a0 * theta0 + b0 * theta1s + c0 * theta1c + d0
    return c_t - C_w * np.cos(alpha) - 0.0059 * np.tan(alpha) * np.cos(alpha) * (mu**2)


def f1(x):
    theta0 = x[0]
    theta1c = x[1]
    theta1s = x[2]
    beta0 = (lock / (w_rf ** 2)) * ((theta0 * (1 + (mu ** 2)) / 8) + (mu * theta1s / 6) - (lamda / 6))
    beta1c = (lock / ((w_rf ** 2) - 1)) * ((theta1c * (1 + 0.5 * (mu ** 2)) / 8) - (mu * beta0 / 6))
    beta1s = (lock / ((w_rf ** 2) - 1)) * ((theta1s * (1 + 1.5 * (mu ** 2)) / 8) + (mu * theta0 / 3) - (lamda * mu / 4) + (beta1c / 8) - (beta1c * (mu ** 2) / 16))
    a1 = (sig * a * 0.5) * ((lamda * mu * 0.5) - (beta1c / 3))
    b1 = (sig * a * 0.5) * ((0.25 * lamda) - (0.25 * mu * beta1c))
    c1 = - sig * a * beta0 / 12
    d1 = (sig * a * 0.5) * ((0.75 * lamda * beta1c) + (beta0 * beta1s / 6) + (0.25 * mu * (beta0**2 + beta1c**2)) + (0.5 * mu * cd0 / a))
    C_H = a1 * theta0 + b1 * theta1s + c1 * theta1c + d1
    C_MxH = (sig * a * 0.5 * beta1s) * (((w_rf ** 2) - 1) / lock)
    return 0.426 * C_H - C_MxH


def f2(x):
    theta0 = x[0]
    theta1c = x[1]
    theta1s = x[2]
    beta0 = (lock / (w_rf ** 2)) * ((theta0 * (1 + (mu ** 2)) / 8) + (mu * theta1s / 6) - (lamda / 6))
    beta1c = (lock / ((w_rf ** 2) - 1)) * ((theta1c * (1 + 0.5 * (mu ** 2)) / 8) - (mu * beta0 / 6))
    beta1s = (lock / ((w_rf ** 2) - 1)) * ((theta1s * (1 + 1.5 * (mu ** 2)) / 8) + (mu * theta0 / 3) - (lamda * mu / 4) + (beta1c / 8) - (beta1c * (mu ** 2) / 16))
    a2 = (sig * a * 0.5) * ((beta0 * mu * 0.75) + (beta1s * (1 + 1.5 * (mu**2)) / 3))
    b2 = (sig * a * 0.5) * ((beta0 * (1 + 3 * (mu**2)) / 6) + (0.5 * mu * beta1s))
    c2 = sig * a * 0.5 * ((0.25 * lamda) + (mu * beta1c * 0.25))
    d2 = (sig * a * 0.5) * ((-1.5 * lamda * mu * beta0) + (beta0 * beta1c * ((1 / 6)-(mu**2))-(0.75 * lamda * beta1c)-(0.25 * mu * (beta1s * beta1c))))
    C_Y = a2 * theta0 + b2 * theta1s + c2 * theta1c + d2
    C_MyH = - (sig * a * 0.5 * beta1c) * (((w_rf ** 2) - 1) / lock)
    return 0.426 * C_Y - C_MyH


def f3(x):
    theta0 = x[0]
    theta1c = x[1]
    theta1s = x[2]
    alpha = x[3]
    beta0 = (lock / (w_rf ** 2)) * ((theta0 * (1 + (mu ** 2)) / 8) + (mu * theta1s / 6) - (lamda / 6))
    beta1c = (lock / ((w_rf ** 2) - 1)) * ((theta1c * (1 + 0.5 * (mu ** 2)) / 8) - (mu * beta0 / 6))
    beta1s = (lock / ((w_rf ** 2) - 1)) * ((theta1s * (1 + 1.5 * (mu ** 2)) / 8) + (mu * theta0 / 3) - (lamda * mu / 4) + (beta1c / 8) - (beta1c * (mu ** 2) / 16))
    a1 = (sig * a * 0.5) * ((lamda * mu * 0.5) + (beta1c / 3))
    b1 = (sig * a * 0.5) * ((0.25 * lamda) + (0.25 * mu * beta1c))
    c1 = sig * a * beta0 / 12
    d1 = (sig * a * 0.5) * ((0.75 * lamda * beta1c) + (beta0 * beta1s / 6) + (0.25 * mu * (beta0 ** 2 + beta1c ** 2)) + (0.5 * mu * cd0 / a))
    C_H = a1 * theta0 + b1 * theta1c + c1 * theta1s + d1
    return C_H - C_w * np.sin(alpha) - 0.0059 * (mu**2) / np.cos(alpha)


def jacobian(F, y):
    n = len(F)
    delta = 0.00125
    idelta = 1/delta
    jacob = []
    for i in range(0, n):
        f = F[i]
        x = y.copy()
        dF = []
        for j in range(0, n):
            x[j] += delta
            df = (f(x) - f(y)) * idelta
            x[j] -= delta
            dF.append(df)
        jacob.append(dF)
    return jacob


def inv_jacobian(F, y):
    jacobian_matrix = jacobian(F, y)
    return inv(jacobian_matrix)


def newton_raphson(F, y):
    iterations = 10 ** 3
    tolerance = 10 ** -5
    damping = 1
    y_old = y.copy()
    y_new = y
    error = 1
    i = 0
    while np.amax(error) >= tolerance and i <= iterations:
        f = []
        for j in range(0, len(F)):
            f.append(F[j](y_old))
        inv_jacob = inv_jacobian(F, y)
        y_new = y_old - damping * np.matmul(inv_jacob, f)
        error = abs(y_new - y_old)
        y_old = y_new
        i += 1
    # print('total iterations: ', i)
    if i < iterations:
        return y_new
    else:
        print('Value did not converge')
        return 1


F = [f0, f1, f2, f3]
y = [0, 0, 0, 0]
trim_state = []

for i in np.arange(0, 0.31, 0.05):
    Rmod.mu = i
    mu = Rmod.mu
    c_t = thrust_coefficient(y)
    Rmod.c_t = c_t
    lamda = Rmod.inflow_calculation()
    theta0 = y[0]
    theta1c = y[1]
    theta1s = y[2]
    print(mu, [beta0, beta1c, beta1s])
    g = newton_raphson(F, y)
    # print(y, g)
    b0 = beta0
    b1c = beta1c
    b1s = beta1s
    beta0 = (lock / (w_rf ** 2)) * ((theta0 * (1 + (mu ** 2)) / 8) + (mu * theta1s / 6) - (lamda / 6))
    beta1c = (lock / ((w_rf ** 2) - 1)) * (((theta1c - b1s) * (1 + 0.5 * (mu ** 2)) / 8) - (mu * beta0 / 6))
    beta1s = (lock / ((w_rf ** 2) - 1)) * ((theta1s * (1 + 1.5 * (mu ** 2)) / 8) + (mu * theta0 / 3) - (lamda * mu / 4) + (b1c / 8) - (b1c * (mu ** 2) / 16))
    trim_state.append([i, lamda, list(g)])
    y = g.copy()
