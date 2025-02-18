import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Datos empíricos
np.random.seed(42)
datos = np.random.uniform(0, 10, 1000)  # Datos empíricos uniformes entre 0 y 10

# Intervalo de interés
x1, x2 = 4, 6

# Estimar la densidad con KDE
kde = gaussian_kde(datos)

# Crear valores para graficar la densidad
x = np.linspace(0, 10, 1000)
densidad = kde(x)

# Valores dentro del intervalo
x_intervalo = np.linspace(x1, x2, 500)
densidad_intervalo = kde(x_intervalo)

# Calcular el área bajo la curva (probabilidad en el intervalo)
area = np.trapz(densidad_intervalo, x_intervalo)

# Graficar la densidad
plt.plot(x, densidad, label='Densidad estimada (KDE)')
plt.fill_between(x_intervalo, densidad_intervalo, alpha=0.5, color='orange', label=f'Área resaltada: {area:.4f}')

# Añadir el valor del área en el gráfico
x_pos_texto = (x1 + x2) / 2  # Punto medio del intervalo
y_pos_texto = kde((x1 + x2) / 2) * 0.5  # Altura aproximada
plt.text(x_pos_texto, y_pos_texto, f'{area:.4f}', color='black', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Etiquetas y leyenda
plt.title('Distribución empírica con área resaltada')
plt.xlabel('X')
plt.ylabel('Densidad estimada')
plt.legend()
plt.grid(alpha=0.3)
plt.show()