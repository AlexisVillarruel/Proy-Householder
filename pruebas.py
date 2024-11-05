import numpy as np
import matplotlib.pyplot as plt
import time
from codigoTriangulacionHouseHolder import householder_reflection, linear_regression_householder

# --- Prueba de Tiempo de Ejecución ---

# Configuración de tamaños de matrices para la prueba
matrix_sizes = [(100, 10), (200, 20), (300, 30), (400, 40), (500, 50)]
times = []

# Función para generar matrices de rango completo y bien condicionadas
def generate_well_conditioned_matrix(m, n, condition_number=10):
    """
    Genera una matriz de tamaño (m, n) que cumple con:
    - Rango completo
    - Número de condición controlado
    """
    # Genera una matriz aleatoria y realiza su descomposición SVD
    U, _, Vt = np.linalg.svd(np.random.rand(m, n), full_matrices=False)
    # Crea un conjunto de valores singulares con un número de condición específico
    singular_values = np.linspace(condition_number, 1, n)
    # Reconstruye la matriz con el rango completo y el número de condición deseado
    A = U @ np.diag(singular_values) @ Vt
    return A

# Prueba de tiempo de ejecución para la descomposición QR de Householder
for m, n in matrix_sizes:
    A = generate_well_conditioned_matrix(m, n)  # Genera una matriz bien condicionada de tamaño (m, n)
    start_time = time.time()   # Marca el tiempo de inicio
    householder_reflection(A)  # Llama a la descomposición QR
    elapsed_time = time.time() - start_time  # Calcula el tiempo de ejecución
    times.append(elapsed_time)  # Guarda el tiempo

# Convertir tamaños de matrices en valores escalares para el eje X
sizes = [m * n for m, n in matrix_sizes]

# Generar gráfico de tiempo de ejecución
plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o', linestyle='-', color='b')
plt.xlabel("Tamaño de la Matriz (m x n)")
plt.ylabel("Tiempo de Ejecución (segundos)")
plt.title("Tiempo de Ejecución de Descomposición QR mediante Householder")
plt.grid(True)
plt.show()

# --- Prueba de Estabilidad Numérica ---

# Configuración de prueba para estabilidad numérica
np.random.seed(0)
m, n = 100, 10
X_base = np.random.rand(m, n)
y_stability = np.random.rand(m)

# Lista de números de condición para observar la estabilidad numérica
condition_numbers = [1, 10, 100, 1e3, 1e4]
mse_values = []

# Prueba de estabilidad numérica variando el número de condición de X
for cond_num in condition_numbers:
    U, _, Vt = np.linalg.svd(X_base, full_matrices=False)  # Descomposición SVD de X_base
    singular_values = np.linspace(cond_num, 1, n)  # Genera valores singulares con el número de condición deseado
    X_cond = U @ np.diag(singular_values) @ Vt  # Reconstruye X con el número de condición deseado
    
    # Calcular beta_hat usando la función de regresión lineal de Householder
    beta_hat, error = linear_regression_householder(X_cond, y_stability)
    
    # Si hay un error (por ejemplo, beta_hat es None), omitir este punto
    if error:
        print(f"Número de condición: {cond_num}, Error: {error}")
        mse_values.append(np.nan)  # Usa NaN para indicar que hubo un error en este caso
    else:
        # Predicción y cálculo del MSE
        y_pred = X_cond @ beta_hat
        mse = np.mean((y_stability - y_pred) ** 2)
        mse_values.append(mse)
        print(f"Número de condición: {cond_num}, MSE: {mse}")

# Generar gráfico de estabilidad numérica
plt.figure(figsize=(10, 6))
plt.plot(condition_numbers, mse_values, marker='o', linestyle='-', color='r')
plt.xscale("log")
plt.xlabel("Número de Condición de la Matriz X (escala logarítmica)")
plt.ylabel("Error Cuadrático Medio (MSE)")
plt.title("Impacto del Número de Condición en la Estabilidad Numérica de Householder QR")
plt.grid(True)
plt.show()
