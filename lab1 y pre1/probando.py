import matplotlib.pyplot as plt
import numpy as np
# 1N4148
# Definir los datos de x, y las incertidumbres:
"""x = np.array([0,10, 20, 28, 36, 40, 44, 44, 44, 48, 48, 48, 48, 52, 52, 52, 52, 52, 56, 56, 56, 56, 56, 60, 60, 60, 60, 60, 60, 64, 64, 64, 64, 64, 64,]) # Datos de x
y = np.array([0,0.009,0.017,0.027,0.036,0.044,0.056,0.064,0.076,0.085,0.095,0.105,0.115,0.135,0.155,0.175,0.195,0.215,0.254,0.274,0.294,0.354,0.394,0.514,0.594,0.714,0.794,0.894,0.994,1.090,1.190,1.290,1.490,1.790,1.990]) # Datos de y
incertidumbres_x = np.array([0,1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,]) # Incertidumbres de x"""
incertidumbres_y = np.array([0,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,0.150,])

# 1N4007
x= np.array([0,100,190,280,360,400,400,440,440,440,440,440,440,480,480,480,480,480,520,520,520,520,520,520,560,560,560,560,560,600,600,600,600,600,600,])
y= np.array([0,0,0,0.002,0.004,0.008,0.02,0.024,0.036,0.046,0.056,0.066,0.076,0.092,0.112,0.132,0.152,0.172,0.208,0.228,0.248,0.308,0.348,0.468,0.544,0.664,0.744,0.844,0.944,1.040,1.140,1.240,1.440,1.740,1.940])
incertidumbres_x = np.array([0,10,10,20,20,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40])

# Graficar los datos utilizando el método errorbar de matplotlib:
plt.errorbar(x, y, xerr=incertidumbres_x, yerr=incertidumbres_y, fmt='o', markersize=0, capsize=4, elinewidth=1)
plt.plot(x, y, 'r-', label='Curva Característica 1N4007', linewidth=2)
plt.xlabel('Vb [mV]') # Etiqueta del eje x
plt.ylabel('I [mA]') # Etiqueta del eje y

plt.xticks(np.arange(0, 1.2*max(x), 40)) #max(x)+1 paso 2 para 1N4148 
plt.yticks(np.arange(0, max(y)+0.2, 0.2)) #min(y)

# Agregar la recta con la ecuación y = -0.1x + 1
x_recta = np.arange(max(x)+ max(x)/4)
y_recta = -0.0001 * x_recta + 1
#plt.plot(x_recta, y_recta, color='green', label='recta de carga')
print(y_recta[60])
#x1 = np.linspace(0, 64, 100)

#recta estática
#plt.plot(x_recta, 1.686*(10**(-3))*x_recta, color='purple', label='modo estático de operación') #16.57*(10**(-3))*x_recta

#recta dinámica gran señal 48, 0.087 * (x_din-60) + 0.994 para 1N4007
"""x_otro=np.arange(0,428)
y_otro=np.zeros(428)
x_din = np.arange(428, 1.10*max(x)) #(2/3)*max(x)
y_din = 0.00715 * (x_din-560) + 0.944
plt.plot(x_din, y_din, color='gray', label='modelo dinámico de gran señal')
plt.plot(x_otro, y_otro, color='gray')"""

#recta dinámica baja señal
x_otro=np.arange(0,560)
y_otro=np.zeros(560)
plt.plot(x_otro, y_otro, color='brown')
x_bs = np.ones(12)*560
y_bs = np.arange(0,2,1/6)
plt.plot(x_bs,y_bs, color='brown', label='modelo dinámico de baja señal')

# Calcular el punto de intersección
y_interseccion = np.intersect1d(y, y_recta)
# Trazar el punto de intersección
plt.plot(560, y_interseccion, 'bo', label=f'punto de operacion (560mV, {y_interseccion[0]}mA)', color='black') #60 para 1N4148

#plt.xlim(0, 100)
#plt.ylim(-0.3, 2.309)

plt.title('Curva experimental diodo 1N4007') # Título del gráfico
plt.grid(True) # Mostrar cuadrícula en el gráfico
#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.legend() # Mostrar leyenda
plt.show() # Mostrar el gráfico




"""import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

# Ajuste de la curva de regresión lineal
coefficients = np.polyfit(x, y, 1)  # Ajuste de una línea recta (polinomio de grado 1)
m = coefficients[0]  # Pendiente
b = coefficients[1]  # Intersección en el eje y

# Generar los valores predichos por la curva de regresión lineal
y_pred = m * x + b

# Graficar los puntos y la curva de regresión lineal
plt.scatter(x, y, color='red', label='Puntos')
plt.plot(x, y_pred, color='blue', label='Regresión lineal')

# Configurar el gráfico
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste de curva de regresión lineal')
plt.legend()

# Mostrar el gráfico
plt.show()"""



"""import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([2, 1.5, 1, 0.5, 0.25, 0.125])

# Ajuste de la curva exponencial
coefficients = np.polyfit(x, np.log(y), 1)  # Ajuste de una curva exponencial (polinomio de grado 1)
a = np.exp(coefficients[1])  # Coeficiente de la curva exponencial
b = coefficients[0]  # Exponente de la curva exponencial

# Generar los valores predichos por la curva exponencial
y_pred = a * np.exp(b * x)

# Graficar los puntos y la curva exponencial
plt.scatter(x, y, color='red', label='Puntos')
plt.plot(x, y_pred, color='blue', label='Curva exponencial')

# Configurar el gráfico
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste de curva exponencial')
plt.legend()

# Mostrar el gráfico
plt.show()"""
