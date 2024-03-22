import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy.abc import x,y,z

class data():
    def __init__(self, abscisas, ordenadas):
        self.x = np.array(abscisas) #.reshape((len(abscisas),1))
        self.y = np.array(ordenadas) #.reshape((len(ordenadas),1))

    def error(self, y, fx):
        error = 0
        for i, j in zip(y, fx):
            error += (i-j)**2
        return error

    def graficar_pol_sp(self, polinomio, titulo):
        x = sp.symbols('x')
        plt.grid()
        plt.plot(self.x,self.y,'bo')

        d = np.linspace(min(self.x),max(self.x),100)
        f = [polinomio.subs(x,i) for i in d]
        plt.plot(d,f,'r-')

        plt.title(titulo)
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.show()

    def graficar_pol_np(self, polinomio, titulo):
        plt.grid()
        plt.plot(self.x,self.y,'bo')

        d = np.linspace(min(self.x),max(self.x),100)
        plt.plot(d,polinomio(d),'r-')

        plt.title(titulo)
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.show()


    def polinomio_coef_ind(self):
        m1 = len(self.x) #, n1
        m2 = len(self.y) #, n2
        if (m1 != m2): # or (n1 != n2):
            raise ValueError('x e y no tienen la misma dimensión')
        
        m = m1
        n = m - 1
        B = np.ones((m, m)) # inicializar una matriz con unos

        # completar las m-1 filas de B, con las potencias apropiadas de los valores
        # que corresponden a la variable independiente.
        for i in range(m):
            for j in range(1, m):
                B[i, j] = self.x[i]**(j)

        c = np.linalg.solve(B, self.y.reshape((m2,1))) # resuelve el sistema de ecuaciones, dando como resultado los coeficientes del polinomio. Cambio de y con-y

        c = np.flip(c).T  #invirtiendo coeficientes (agregado)

        p = np.poly1d(c[0]) # librería para trabajo simbólico de polinomios

        return c, p

    def polinomio_lagrange(self):
        x = sp.symbols('x')
        
        n = len(self.x)
        p = 0
        for k in range(n):
            d = 1
            for i in range(n):
                if i != k:
                    d = d * (x - self.x[i]) / (self.x[k] - self.x[i])
            p += self.y[k] * d

            p = sp.simplify(p) #simplificando a su minima expresion el polinomio de lagrage
        return p

    def polinomio_newton(self):
        if len(self.x) != len(self.y):
            raise ValueError('Error en el tamaño de la data')    

        n = len(self.x)
        F = [[0] * n for _ in range(n)]

        for i in range(n):
            F[i][0] = self.y[i]

        for i in range(1, n):
            for j in range(1, i + 1):
                F[i][j] = (F[i][j - 1] - F[i - 1][j - 1]) / (self.x[i] - self.x[i - j])

        coeficientes = [F[i][i] for i in range(n)]

        #creando polinomio
        ecuacion = 0
        termino = 1
        var = sp.symbols('x')
        for k in range(n):
            ecuacion += coeficientes[k]*termino
            termino *= (var - self.x[k])
            
        p = sp.simplify(ecuacion)

        return F, p

    def polinomio_min_cuad(self, G):
        n = len(self.x)
        B = np.zeros(G + 1)
        F = np.zeros((n, G + 1))

        for k in range(1, G + 2):
            F[:, k - 1] = self.x ** (k - 1)

        A = np.dot(F.T, F)
        B = np.dot(F.T, self.y)
        C = np.linalg.solve(A, B)
        
        c = np.flip(C).T  #invirtiendo coeficientes (agregado)
        #polinomio
        p = np.poly1d(c)

        return C, p

    def __coeficientes_poli_trig(self, G):
        X = np.linspace(0,2*np.pi,len(self.x)) #copias
        Y = self.y.copy()
        
        n = len(X) - 1
        maxg = int((n - 1) / 2)
        if G > maxg:
            G = maxg

        A = np.zeros(G + 1)
        B = np.zeros(G + 1)
        Yes = (Y[0] + Y[n]) / 2
        Y[0] = Yes
        Y[n] = Yes
        A[0] = np.sum(Y)

        for i in range(1, G + 1):
            A[i] = np.dot(np.cos(i * X), Y.T)
            B[i] = np.dot(np.sin(i * X), Y.T)

        A = 2 * A / n
        B = 2 * B / n
        A[0] = A[0] / 2

        return A, B

    def polinomio_trigonometrico(self, G):
        A, B = self.__coeficientes_poli_trig(G)
        var = (2*sp.pi/max(self.x))*x
        f = A[0]
        for k in range(1, G + 1):
            f += A[k] * sp.cos(k * var) + B[k] * sp.sin(k * var)

        return f