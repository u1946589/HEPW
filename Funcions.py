#PER NO COPIAR TOTES LES FUNCIONS A L'ARXIU PRINCIPAL. IMPORTAR-LES
import numpy as np
import numba as nb
from mpmath import mp  # per tenir més decimals
mp.dps = 50
from numba import jit
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse import lil_matrix, diags, hstack, vstack
from scipy.sparse.linalg import spsolve, factorized
from numpy import zeros, ones, mod, conj, array, r_, linalg, Inf, complex128, c_, r_, angle

#@nb.jit

def pade4all(order, coeff_mat, s):

    nbus = coeff_mat.shape[1]
    voltages = np.zeros(nbus, dtype=complex)
    if order % 2 != 0:
        nn = int(order / 2)
        L = nn
        M = nn
        for d in range(nbus):
            rhs = coeff_mat[L + 1:L + M + 1, d]
            C = np.zeros((M, M), dtype=complex)
            for i in range(M):
                k = i + 1
                C[i, :] = coeff_mat[L - M + k:L + k, d]
            b = np.zeros(rhs.shape[0] + 1, dtype=complex)
            x = np.linalg.solve(C, -rhs)  # bn to b1
            b[0] = 1
            b[1:] = x[::-1]
            a = np.zeros(L + 1, dtype=complex)
            a[0] = coeff_mat[0, d]
            for i in range(L):
                val = complex(0)
                k = i + 1
                for j in range(k + 1):
                    val += coeff_mat[k - j, d] * b[j]
                a[i + 1] = val
            p = complex(0)
            q = complex(0)

            for i in range(len(a)):
                p += a[i] * s ** i
            for i in range(len(b)):
                q += b[i] * s ** i
            voltages[d] = p / q
            #ppb = np.poly1d(b)
            #ppa = np.poly1d(a)
            #ppbr = ppb.r  # arrels, o sigui, pols
            #ppar = ppa.r  # arrels, o sigui, zeros
    else:
        nn = int(order / 2)
        L = nn
        M = nn - 1
        for d in range(nbus):
            rhs = coeff_mat[M + 2: 2 * M + 2, d]
            C = np.zeros((M, M), dtype=complex)
            for i in range(M):
                k = i + 1
                C[i, :] = coeff_mat[L - M + k:L + k, d]
            b = np.zeros(rhs.shape[0] + 1, dtype=complex)
            x = np.linalg.solve(C, -rhs)  # de bn a b1, en aquest ordre
            b[0] = 1
            b[1:] = x[::-1]
            a = np.zeros(L + 1, dtype=complex)
            a[0] = coeff_mat[0, d]
            for i in range(1, L):
                val = complex(0)
                for j in range(i + 1):
                    val += coeff_mat[i - j, d] * b[j]
                a[i] = val
            val = complex(0)
            for j in range(L):
                val += coeff_mat[M - j + 1, d] * b[j]
            a[L] = val
            p = complex(0)
            q = complex(0)

            for i in range(len(a)):
                p += a[i] * s ** i
            for i in range(len(b)):
                q += b[i] * s ** i
            voltages[d] = p / q
    return voltages



@nb.jit
def eta(U_inicial, limit):
    complex_type = nb.complex128
    n = limit
    Um = np.zeros(n, complex_type)
    Um[:] = U_inicial[:limit]
    mat = np.zeros((n, n+1), complex_type)
    mat[:, 0] = np.inf  # infinit
    mat[:, 1] = Um[:]
    for j in range(2, n + 1):
        if j % 2 == 0:
            for i in range(0, n + 1 - j):
                mat[i, j] = 1 / (1 / mat[i+1, j-2] + 1 / (mat[i+1, j-1]) - 1 / (mat[i, j-1]))
        else:
            for i in range(0, n + 1 - j):
                mat[i, j] = mat[i+1, j-2] + mat[i+1, j-1] - mat[i, j-1]
    return np.sum(mat[0, 1:])

@nb.jit
def aitken(U, limit):
    def S(Um, k):
        suma = 0
        for m in range(k+1):
            suma += Um[m]
        return suma
    complex_type = nb.complex128
    Um = U[:limit]  # només els 10 primers coeficients, si no, divideix per 0 i es deteriora
    n = limit
    T = np.zeros(n-2, complex_type)
    for i in range(len(T)):
        T[i] = S(Um, i+2) - (S(Um, i+1) - S(Um, i))**2 / ((S(Um, i+2) - S(Um, i+1)) - (S(Um, i+1)-S(Um, i)))

    return T[-1]  # l'últim element, entenent que és el que millor aproxima

@nb.jit
def shanks(U, limit):
    def S(Um, k):
        suma = 0
        for m in range(k+1):
            suma += Um[m]
        return suma
    complex_type = nb.complex128
    Um = U[:limit]  # només els 10 primers coeficients, si no, divideix per 0 i es deteriora
    n = limit
    n_trans = 3
    T = np.zeros((n, n_trans), complex_type)
    for lk in range(n_trans):
        for i in range(n-2*lk):
            if lk == 0:
                T[i, lk] = S(Um, i+2) - (S(Um, i+1) - S(Um, i))**2 / ((S(Um, i+2) - S(Um, i+1)) - (S(Um, i+1)-S(Um, i)))
            else:
                #T[i, lk] = S(T[:, lk-1], i+2) - (S(T[:, lk-1], i+1) - S(T[:, lk-1], i))**2 / \
                           #((S(T[:, lk-1], i+2) - S(T[:, lk-1], i+1)) - (S(T[:, lk-1], i+1)-S(T[:, lk-1], i)))
                T[i, lk] = T[i+2, lk - 1] - (T[i+2, lk-1]-T[i+1, lk-1])**2 / \
                           ((T[i+2, lk-1]-T[i+1, lk-1]) - (T[i+1, lk-1]-T[i, lk-1]))

    return T[n-2*(n_trans-1) -1, n_trans-1]  # l'últim element, entenent que és el que millor aproxima

@nb.jit
def theta(U_inicial, limit):
    def S(Um, k):
        suma = 0
        for m in range(k+1):
            suma = suma + Um[m]
        return suma
    complex_type = nb.complex128
    n = limit
    Um = np.zeros(n, complex_type)
    Um[:] = U_inicial[:limit]
    mat = np.zeros((n, n+1), complex_type)
    for i in range(n):
        mat[i, 1] = S(Um, i)  # plena de sumes parcials
    for j in range(2, n+1):
        if j % 2 == 0:
            for i in range(0, n+1-j):
                mat[i, j] = mat[i+1, j-2] + 1 / (mat[i+1, j-1] - mat[i, j-1])
        else:
            for i in range(0, n + 1 - j):
                mat[i, j] = mat[i+1, j-2] + ((mat[i+2, j-2] - mat[i+1, j-2]) * (mat[i+2, j-1] - mat[i+1, j-1])) \
                            / (mat[i+2, j-1] - 2 * mat[i+1, j-1] + mat[i, j-1])
    if limit % 2 == 0:
        return mat[0, n-1]
    else:
        return mat[0, n]

"""
@nb.jit
def theta(U_inicial, limit):
    def S(Um, k):
        suma = 0
        for m in range(k+1):
            suma = suma + Um[m]
        return suma
    complex_type = nb.complex128
    n = limit
    Um = np.zeros(n, complex_type)
    Um[:] = U_inicial[:limit]
    mat = np.zeros((n, n+1), complex_type)
    for i in range(n):
        mat[i, 1] = S(Um, i)  # plena de sumes parcials

    #col_perd = int((n-2)/2) - 1
    col_perd = int((n - 1) / 2) - 1

    for j in range(2, n+1-col_perd):
        if j % 2 == 0:
            for i in range(n+1-j-int((j-1)/2)):
                mat[i, j] = mat[i+1, j-2] + 1 / (mat[i+1, j-1] - mat[i, j-1])
        else:
            for i in range(n+1-j-int((j-1)/2)):
                mat[i, j] = mat[i+1, j-2] + ((mat[i+2, j-2] - mat[i+1, j-2]) * (mat[i+2, j-1] - mat[i+1, j-1])) / ((mat[i+2, j-1] - mat[i+1, j-1]) - (mat[i+1, j-1] - mat[i, j-1]))

    #print(mat)
    if n % 2 == 0 and col_perd % 2 == 0:
        #print(mat[4, n - col_perd - 3])
        return mat[4, n - col_perd - 3]
    elif n % 2 == 0 and col_perd % 2 != 0:
        #print(mat[2, n - col_perd - 2])
        return mat[2, n - col_perd - 2]
    elif n % 2 != 0 and col_perd % 2 == 0:
        #print(mat[3, n - col_perd - 2])
        return mat[3, n - col_perd - 2]
    else:
        #print(mat[1, n - col_perd - 1])
        return mat[1, n - col_perd - 1]
"""
@nb.jit
def rho(U, limit):  # veure si cal tallar U, o sigui, agafar per exemple els 10 primers coeficients
    def S(Um, k):
        suma = 0
        for m in range(k+1):
            suma = suma + Um[m]
        return suma
    complex_type = nb.complex128
    Um = U[:limit]  # no agafar tots els coeficients, si no, salta error
    n = limit
    mat = np.zeros((n, n+1), complex_type)
    for i in range(n):
        mat[i, 1] = S(Um, i)  # plena de sumes parcials
    for j in range(2, n+1):
        for i in range(0, n+1-j):
            mat[i, j] = mat[i+1, j-2] + (j - 1) / (mat[i+1,j-1] - mat[i, j-1])
    if limit % 2 == 0:
        return mat[0, n-1]
    else:
        return mat[0, n]


@nb.jit
def epsilon2(U, limit):
    def S(Um, k):
        suma = 0
        for m in range(k+1):
            suma = suma + Um[m]
        return suma
    complex_type = nb.complex128
    Um = U[:limit]  # no agafar tots els coeficients, si no, salta error
    n = limit
    mat = np.zeros((n, n+1), complex_type)
    for i in range(n):
        mat[i, 1] = S(Um, i)  # plena de sumes parcials
    for j in range(2, n+1):
        for i in range(0, n+1-j):
            mat[i, j] = mat[i+1, j-2] + 1 / (mat[i+1, j-1] - mat[i, j-1])
    if limit % 2 == 0:
        return mat[0, n-1]
    else:
        return mat[0, n]

@nb.jit
def thevenin_funcX2(U, X, i):
    complex_type = nb.complex128
    n = len(U)
    r_3 = np. zeros(n, complex_type)
    r_2 = np. zeros(n, complex_type)
    r_1 = np. zeros(n, complex_type)
    r_0 = np. zeros(n, complex_type)
    T_03 = np. zeros(n, complex_type)
    T_02 = np. zeros(n, complex_type)
    T_01 = np. zeros(n, complex_type)
    T_00 = np. zeros(n, complex_type)
    T_13 = np. zeros(n, complex_type)
    T_12 = np. zeros(n, complex_type)
    T_11 = np. zeros(n, complex_type)
    T_10 = np. zeros(n, complex_type)
    T_23 = np. zeros(n, complex_type)
    T_22 = np. zeros(n, complex_type)
    T_21 = np. zeros(n, complex_type)
    T_20 = np. zeros(n, complex_type)

    #A LA NOVA MANERA, CONSIDERANT QUE U[0] POT SER DIFERENT D'1:
    r_0[0] = -1
    r_1[0:n - 1] = U[1:n] / U[0]
    r_2[0:n - 2] = U[2:n] / U[0] - U[1] * np.conj(U[0]) / U[0] * X[1:n - 1]

    T_00[0] = -1
    T_01[0] = -1
    T_02[0] = -1
    T_10[0] = 0
    T_11[0] = 1 / U[0]
    T_12[0] = 1 / U[0]
    T_20[0] = 0
    T_21[0] = 0
    T_22[0] = -U[1] * np.conj(U[0]) / U[0]


    for l in range(n):  # ANAR CALCULANT CONSTANTS , RESIDUS I POLINOMIS
        a = (r_2[0] * r_1[0]) / (- r_0[1] * r_1[0] + r_0[0] * r_1[1] - r_0[0] * r_2[0])
        b = -a * r_0[0] / r_1[0]
        c = 1 - b
        T_03[0] = b * T_01[0] + c * T_02[0]
        T_03[1:n] = a * T_00[0:n-1] + b * T_01[1:n] + c * T_02[1:n]
        T_13[0] = b * T_11[0] + c * T_12[0]
        T_13[1:n] = a * T_10[0:n-1] + b * T_11[1:n] + c * T_12[1:n]
        T_23[0] = b * T_21[0] + c * T_22[0]
        T_23[1:n] = a * T_20[0:n-1] + b * T_21[1:n] + c * T_22[1:n]
        r_3[0:n-2] = a * r_0[2:n] + b * r_1[2:n] + c * r_2[1:n-1]

        if l == n - 1:
            t_0 = T_03
            t_1 = T_13
            t_2 = T_23

        r_0[:] = r_1[:]
        r_1[:] = r_2[:]
        r_2[:] = r_3[:]
        T_00[:] = T_01[:]
        T_01[:] = T_02[:]
        T_02[:] = T_03[:]
        T_10[:] = T_11[:]
        T_11[:] = T_12[:]
        T_12[:] = T_13[:]
        T_20[:] = T_21[:]
        T_21[:] = T_22[:]
        T_22[:] = T_23[:]

        r_3 = np.zeros(n, complex_type)
        T_03 = np.zeros(n, complex_type)
        T_13 = np.zeros(n, complex_type)
        T_23 = np.zeros(n, complex_type)

    usw = -np.sum(t_0) / np.sum(t_1)
    sth = -np.sum(t_2) / np.sum(t_1)

    sigma_bo = sth / (usw * np.conj(usw))

    #u = 0.5 + np.sqrt(0.25 + np.real(sigma_bo) - np. imag(sigma_bo)**2) + np.imag(sigma_bo)*1j  # positive branch
    u = 0.5 - np.sqrt(0.25 + np.real(sigma_bo) - np.imag(sigma_bo) ** 2) + np.imag(sigma_bo) * 1j  # negative branch

    ufinal = u*usw

    return ufinal


def Sigma_funcO(coeff_matU, coeff_matX, order, V_slack):
    """
    :param coeff_matU: array with voltage coefficients
    :param coeff_matX: array with inverse conjugated voltage coefficients
    :param order: should be prof - 1
    :param V_slack: slack bus voltage vector. Must contain only 1 slack bus
    :return: sigma complex value
    """
    #complex_type = nb.complex128
    if len(V_slack) > 1:
        print('Sigma values may not be correct')
    V0 = V_slack[0]
    coeff_matU = coeff_matU / V0
    coeff_matX = coeff_matX / V0
    nbus = coeff_matU.shape[1]
    sigmes = np.zeros(nbus, dtype=complex)
    if order % 2 == 0:
        M = int(order / 2) - 1
    else:
        M = int(order / 2)
    for d in range(nbus):
        a = coeff_matU[1:2 * M + 2, d]
        b = coeff_matX[0:2 * M + 1, d]
        C = np.zeros((2 * M + 1, 2 * M + 1), dtype=complex)
        for i in range(2 * M + 1):
            if i < M:
                C[1 + i:, i] = a[:2 * M - i]
            else:
                C[i - M:, i] = - b[:3 * M - i + 1]
        lhs = np.linalg.solve(C, -a)
        sigmes[d] = np.sum(lhs[M:])/(np.sum(lhs[:M]) + 1)
    return sigmes

def SigmaX(coeff_matU, coeff_matX, order, V_slack):
    # pensada per la formulació original, o per la pròpia quan V[0]=1 sempre
    """
    :param coeff_matU: array with voltage coefficients
    :param order: should be prof - 1
    :param V_slack: slack bus voltage vector. Must contain only 1 slack bus
    :return: sigma complex value
    """
    #complex_type = nb.complex128
    if len(V_slack) > 1:
        print('Sigma values may not be correct')
    V0 = V_slack[0]
    coeff_A = np.copy(coeff_matU)
    coeff_B= np.copy(coeff_matX)
    coeff_A[0, :] = 1
    for i in range(1, coeff_matU.shape[0]):
        coeff_A[i, :] = coeff_matU[i, :] - (V0 - 1) * coeff_A[i-1, :]
    coeff_B[0, :] = 1
    for i in range(1, coeff_matX.shape[0]):
        coeff_B[i, :] = coeff_matX[i, :] + (V0 - 1) * coeff_matX[i-1, :]

    nbus = coeff_matU.shape[1]
    sigmes = np.zeros(nbus, dtype=complex)
    if order % 2 == 0:
        M = int(order / 2) - 1
    else:
        M = int(order / 2)
    for d in range(nbus):
        a = coeff_A[1:2 * M + 2, d]
        b = coeff_B[0:2 * M + 1, d]
        C = np.zeros((2 * M + 1, 2 * M + 1), dtype=complex)
        for i in range(2 * M + 1):
            if i < M:
                C[1 + i:, i] = a[:2 * M - i]
            else:
                C[i - M:, i] = - b[:3 * M - i + 1]
        lhs = np.linalg.solve(C, -a)
        sigmes[d] = np.sum(lhs[M:])/(np.sum(lhs[:M]) + 1)
    return sigmes