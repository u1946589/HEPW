import numpy as np
from mpmath import *

n =100
Z1 =0.1+0.5*1j
Z2 =0.02+0.13*1j
Z3 =0.023+0.1*1j
Zp = -10*1j
Y1 =1/ Z1
Y2 =1/ Z2
Y3 =1/ Z3
Yp =1/ Zp
P=-1
Q= -0.1
Va =1.1
Vb=np. zeros (n, dtype = complex )
Vc=np. zeros (n, dtype = complex )
R=np. zeros (n, dtype = complex )
X=np. zeros (n, dtype = complex )
F=np. zeros (n, dtype = complex )
L=np. zeros (n, dtype = complex )
Y=np. zeros (n, dtype = complex )
M=np. zeros (n, dtype = complex )
B=np. zeros (n, dtype = complex )
INL=np. zeros (n, dtype = complex )
van =0.5
lam =2* np. sqrt (2)/np.pi
In=np. sqrt(1- van*van *(2 - lam*lam)) * 1
ang=-np.pi /2+ np. arctan (( van*np. sqrt ( lam *lam -van *van ))/(1 - van *van ))
Vb [0]= Va
Vc [0]=( - Va*Y1 -Vb [0]* Y3)/(-Y1 -Y3)
R [0]=1/ conj (Vb [0])
X [0]=1/ np. real (Vc [0])
F [0]= np. imag (Vc [0]) *X[0]
B [0]=1+ F [0]* F[0]
L [0]= np. sqrt (B [0])
Y [0]=1/ L[0]
M [0]= F [0]* Y[0]
INL [0]= In *1*( cos( ang)*Y[0] - sin( ang)*M [0]) +In *1*( sin (ang )*Y [0]+ cos (ang )*M [0]) *1j

def sumaR (R,Vb ,i):
    suma =0
    for k in range(i):
        suma = suma +R[k]* conj (Vb[i-k])
    return suma
def sumaX (X,Vc ,i):
    suma =0
    for k in range(i):
        suma = suma +X[k]* np. real (Vc[i-k])
    return suma

def sumaF (Vc ,X,i):
    suma =0
    for k in range(i+1) :
        suma = suma +np. imag (Vc[k])*X[i-k]
    return suma

def sumaY (Y,L,i):
    suma =0
    for k in range(i):
        suma = suma +Y[k]*L[i-k]
    return suma
def sumaM (F,Y,i):
    suma =0
    for k in range(i+1) :
        suma = suma +F[k]*Y[i-k]
    return suma

def sumaB (F,i):
    suma =0
    for k in range(i+1) :
        suma = suma +F[k]*F[i-k]
    return suma

def sumaL (L,i):
    suma =0
    for k in range(1,i):
        suma = suma + L[k] * L[i - k]
    return suma
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


for i in range(1, n):
    Vb[i] = ((P - Q * 1j) * R[i - 1] - Y3 * (Vb[i - 1] - Vc[i - 1]) - Vb[i - 1] * Yp) / Y2
    Vc[i] = (INL[i - 1] - Vb[i] * Y3) / (-Y1 - Y3)
    R[i] = - sumaR(R, Vb, i) / conj(Vb[0])
    X[i] = - sumaX(X, Vc, i) / np.real(Vc[0])
    F[i] = sumaF(Vc, X, i)
    B[i] = sumaB(F, i)
    L[i] = (B[i] - sumaL(L, i)) / (2 * L[0])  # canviat !
    Y[i] = - sumaY(Y, L, i) / L[0]
    M[i] = sumaM(F, Y, i)
    INL[i] = In * (cos(ang) * Y[i] - sin(ang) * M[i]) + In * (sin(ang) * Y[i] + cos(ang) * M[i]) * 1j

    #I1 = (Va - np.sum(Vc)) / Z1
    #I2 = (Va - np.sum(Vb)) / Z2
    #I3 = (np.sum(Vb) - np.sum(Vc)) / Z3
    #Ipq = conj((-P - Q * 1j) / (np.sum(Vb)))
    #Ip = (np.sum(Vb)) / Zp

    I1 = (Va - pade4all(i, Vc, 1)) / Z1
    I2 = (Va - pade4all(i, Vb, 1)) / Z2
    I3 = (pade4all(i, Vb, 1) - pade4all(i, Vc, 1)) / Z3
    Ipq = conj((-P - Q * 1j) / (pade4all(i, Vb, 1)))
    Ip = (pade4all(i, Vb, 1)) / Zp

    sumatori1 = I2 - (Ip + I3 + Ipq)
    sumatori2 = I1 + I3 - np.sum(INL)
    print(abs(sumatori1))

#print(np.sum(INL))
#print(np.sum(Vc))
#print(np.sum(Vb))

angvc = np.angle(np.sum(Vc))

for h in (3, 5, 7, 9):
    Z1 = 0.1 + 0.5 * 1j * h
    Z2 = 0.02 + 0.13 * 1j * h
    Z3 = 0.023 + 0.1 * 1j * h
    Zp = -10 * 1j / h
    Ih = 2*np.sqrt(2)*0.5/(np.pi*h**2)
    angh = h*np.arcsin(0.5/h) + angvc

    Yadm = [[1/Zp+1/Z1+1/Z2, -1/Z2, -1/Z1], [-1/Z2, 1/Z2+1/Z3+1/Zp, -1/Z3], [-1/Z1, -1/Z3, 1/Z1+1/Z3]]
    Yadm = np.array(Yadm)
    Ivec = [[0], [0], [-(Ih*np.cos(angh)+Ih*np.sin(angh)*1j)]]
    Ivec = np.array(Ivec)
    Vsol = np.dot(np.linalg.inv(Yadm), Ivec)
    #print(np.angle(Vsol[2   ])*180/np.pi)





