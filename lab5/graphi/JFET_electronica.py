import sys
sys.path.append('C\\Users\\Brayan Acosta\\Documents\\Semestre 2023-3\\Calculo Numerico\\Drive-Ric\\LAB-03')

import sisEcuNoLin as sis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

"""from interpolante import data
from sympy.abc import x"""

df = pd.read_csv('C:\\Users\\Brayan Acosta\\Documents\\elec_inf_latex\\lab5\\graphi\\Practica5.csv')
datos = np.array(df)

c=0
for k in datos[:,4:7]:
    print(k)
    #buscando coeficientes de la curva
    var = sp.symbols('x1, x2')
    f1 = var[0]*sp.log(k[0]*var[1]) - k[1]
    f2 = var[0]*sp.log((1/5)*var[1]) - 0.8*k[1]
    if k[0] < 1:
        f2 = var[0]*sp.log(0.1*k[0]*var[1]) - 0.8*k[1]
    F = [f1, f2]
    x0 = [0.0001, 2000]
    itera = 50
    tol = 1e-12

    coe = sis.SENL(x0, F, itera, tol)
    print('a: ', coe[0])
    print('b: ', coe[1])

    F0 = sp.lambdify([i for i in var], sp.matrices.Matrix(F))
    print(F0(coe[0],coe[1]))

    #interpolando

    """inter = data([0,0.5*k[0],k[0],1.5*k[0],20*k[0]], [0,0.5*k[1],k[1],1.5*k[1],1.75*k[1]])
    coe, pol = inter.polinomio_newton()
    pol0 = sp.lambdify(x, pol)"""

    #graficando

    plt.grid(True)
    plt.title(f'Punto de operación en {k[2]}')
    plt.ylabel('ID [A]')
    plt.xlabel('VDS [V]')

    v = np.linspace(0.0000001, 12.1, 500)
    i = coe[0]*np.log(coe[1]*v)
    plt.plot(v, i, 'm-', label='curva característica')
    plt.plot(k[0], k[1], '+', color='black')
    plt.text(k[0], k[1], f'Q({k[0]} , {k[1]:.5f})')
    #plt.plot(1/5, 0.8*k[1], 'ro')

    vr = np.linspace(0.0000001, 12.1, 500)
    if k[0]==12:
        ir = np.zeros(len(vr))
    else:
        ir = (k[1]/(k[0]-12))*(vr - 12)
    plt.plot(vr, ir,'b-', label="recta de carga")

    plt.legend()
    plt.show()

    print(c)
    c+=1

