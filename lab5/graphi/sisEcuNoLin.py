import numpy as np
import sympy as sp

def jacobiano(F, var):
    J = sp.zeros(len(F),len(var))
    for i, fi in enumerate(F):
        for j, s in enumerate(var):
            J[i,j] = sp.diff(fi, s)
    return J

def SENL(x0, F, itera, tol):
    x0 = np.array(x0)

    if len(x0) != len(F):
        return 'numero de ecuaciones y variables diferentes, el sistema no se puede resolver'
    
    letras = ''
    for i in range(len(x0)):
        letras += f'x{i+1} '
    var = sp.symbols(letras)

    F = sp.matrices.Matrix(F)

    F0 = sp.lambdify([i for i in var], F, 'numpy')

    J = jacobiano(F, var)

    J0 = sp.lambdify([i for i in var], J, 'numpy')

    k = 1
    while k <= itera:

        try: #sistema con tres variables
            F_x = F0(x0[0], x0[1], x0[2]) #agregar x[n] donde n es el numero de elementos de x0
            J_x = J0(x0[0], x0[1], x0[2])
        except: #sistema con dos variables
            F_x = F0(x0[0], x0[1]) #agregar x[n] donde n es el numero de elementos de x0
            J_x = J0(x0[0], x0[1])

        y = np.linalg.solve(J_x, -F_x).reshape(1,len(x0))[0]
        x0 = x0 + y
        #print(x0)
        if np.linalg.norm(y) < tol:
            return x0
        k += 1
    print('Número máximo de iteraciones excedido')

    return x0

"""var = sp.symbols('x1, x2')
f1 = var[0]*sp.log(8*var[1]) - 5.882353*(10**(-3))
f2 = var[0]*sp.log((1/3)*var[1]) - 5.882353*0.7*(10**(-3))
F = [f1, f2]
x0 = [5, 20]
itera = 50
tol = 1e-12

coe = SENL(x0, F, itera, tol)
print('a', coe[0])
print('b', coe[1])

F0 = sp.lambdify([i for i in var], sp.matrices.Matrix(F))
print(F0(coe[0],coe[1]))"""