import numpy as np
import math
import matplotlib.pyplot as plt

U = 5  # equival a l'E
R = 2  #equival a R_1
R2 = 3
P = 1.2
Vt = 0.026
Is = 0.000005
n = 200 #profunditat arbitrària
Vd = np.zeros(n)
Vl = np.zeros(n)
I1 = np.zeros(n)
I2 = np.zeros(n)
Vx = np.zeros(n)
I1[0] = U/R
Vd[0] = Vt*math.log(1+I1[0]/Is)
Vl[0] = P/I1[0]
def convVd(Vd, I, i):  #funció de convolució pel càlcul de la Vd[i]
    suma = 0
    for k in range(1, i):
        suma = suma+k*Vd[k]*I[i-k]
    return suma
def sumI(I, i):
    suma = 0
    for k in range(0, i+1):
        suma = suma+I[k]
    return suma
def sumV(V, i):
    suma = 0
    for j in range(0, i):
        suma = suma+V[j]
    return suma
def convVlI(Vl, I1, i):  #funció de convolució pel càlcul de la Vl[i]
    suma = 0
    for k in range(i):
        suma = suma + Vl[k]*I1[i-k]
    return suma

for i in range(1, n): #càlcul de termes diferents de 0
    I1[i] = (1/R+1/R2)*(-Vd[i-1]-Vl[i-1])
    Vd[i] = (i*Vt*I1[i]-convVd(Vd, I1, i))/(i*(Is+I1[0]))
    Vl[i] = -convVlI(Vl, I1, i) / I1[0]

print('Ara els resultats finals:')
If = sum(I1)
Vdf = sum(Vd)
Vlf = sum(Vl)
print('I1: '+str(If))
print('Vd: '+str(Vdf))
print('Vl: '+str(Vlf))
print('P: '+str(Vlf*If))
Vdfinal=np.zeros(n)
for j in range(n): #per tal de veure com evoluciona la tensió del díode
    suma=0
    for k in range(j+1):
        suma=suma+Vd[k]
    Vdfinal[j] = suma
print(Vdfinal)



def pade4all(order, coeff_mat, s):

    if order % 2 != 0:
        nn = int(order / 2)
        L = nn
        M = nn
        rhs = coeff_mat[L + 1:L + M + 1]
        C = np.zeros((M, M), dtype=complex)
        for i in range(M):
            k = i + 1
            C[i, :] = coeff_mat[L - M + k:L + k]
        b = np.zeros(rhs.shape[0] + 1, dtype=complex)
        x = np.linalg.solve(C, -rhs)  # bn to b1
        b[0] = 1
        b[1:] = x[::-1]
        a = np.zeros(L + 1, dtype=complex)
        a[0] = coeff_mat[0]
        for i in range(L):
            val = complex(0)
            k = i + 1
            for j in range(k + 1):
                val += coeff_mat[k - j] * b[j]
            a[i + 1] = val
        p = complex(0)
        q = complex(0)

        for i in range(len(a)):
            p += a[i] * s ** i
        for i in range(len(b)):
            q += b[i] * s ** i
        return p/q

            #ppb = np.poly1d(b)
            #ppa = np.poly1d(a)
            #ppbr = ppb.r  # arrels, o sigui, pols
            #ppar = ppa.r  # arrels, o sigui, zeros
    else:
        nn = int(order / 2)
        L = nn
        M = nn - 1

        rhs = coeff_mat[M + 2: 2 * M + 2]
        C = np.zeros((M, M), dtype=complex)
        for i in range(M):
            k = i + 1
            C[i, :] = coeff_mat[L - M + k:L + k]
        b = np.zeros(rhs.shape[0] + 1, dtype=complex)
        x = np.linalg.solve(C, -rhs)  # de bn a b1, en aquest ordre
        b[0] = 1
        b[1:] = x[::-1]
        a = np.zeros(L + 1, dtype=complex)
        a[0] = coeff_mat[0]
        for i in range(1, L):
            val = complex(0)
            for j in range(i + 1):
                val += coeff_mat[i - j] * b[j]
            a[i] = val
        val = complex(0)
        for j in range(L):
            val += coeff_mat[M - j + 1] * b[j]
        a[L] = val
        p = complex(0)
        q = complex(0)

        for i in range(len(a)):
            p += a[i] * s ** i
        for i in range(len(b)):
            q += b[i] * s ** i
        return p/q

Vpa = pade4all(15, Vd, 1)

print(Vpa)

"""
#DOMB-SYKES PEL DÍODE
bb = np. zeros(n, dtype=complex)
for i in range(3, len(Vd) - 1):
    #bb[i, j] = np. abs(np.sqrt((U[i+1, j] * U[i-1, j] - U[i, j] ** 2) / (U[i, j] * U[i-2, j] - U[i-1, j] ** 2)))
    bb[i] = (I1[i]) / (I1[i-1])

vec_1n = np. zeros(n)
for i in range(3, n):
    #vec_1n[i] = 1 / i
    vec_1n[i] = i

plt.plot(vec_1n[3:len(Vd)-1], abs(bb[3:len(Vd)-1]), 'ro ', markersize=2)
plt.show()

for i in range(n):
    print(abs(bb[i]))
"""

for i in range(1, len(Vdfinal)):
    print(abs(pade4all(i, Vd, 1)))
