import time
import matplotlib.pyplot as plt
from Trabajo_Final import solve_cholesky, linear_regression_householder, gram_schmidt_linear_regression, regresion_SVD

def ejecutar_pruebas(X, y):
    tiempos = {}

    # Medir tiempo para la descomposición de Cholesky
    inicio = time.time()
    try:
        beta_cholesky = solve_cholesky(X, y)
        tiempos["Cholesky"] = time.time() - inicio
        print("Coeficientes obtenidos (Cholesky):", beta_cholesky)
    except ValueError as e:
        tiempos["Cholesky"] = None
        print("Error en descomposición de Cholesky:", e)

    # Medir tiempo para la descomposición QR usando Householder
    inicio = time.time()
    try:
        beta_householder, error = linear_regression_householder(X, y)
        tiempos["Householder"] = time.time() - inicio
        if error:
            print("Error en descomposición QR Householder:", error)
        else:
            print("Coeficientes obtenidos (Householder):", beta_householder)
    except ValueError as e:
        tiempos["Householder"] = None
        print("Error en descomposición QR Householder:", e)

    # Medir tiempo para la ortogonalización de Gram-Schmidt
    inicio = time.time()
    try:
        beta_gram_schmidt = gram_schmidt_linear_regression(X, y)
        tiempos["Gram-Schmidt"] = time.time() - inicio
        print("Coeficientes obtenidos (Gram-Schmidt):", beta_gram_schmidt)
    except ValueError as e:
        tiempos["Gram-Schmidt"] = None
        print("Error en ortogonalización de Gram-Schmidt:", e)

    # Medir tiempo para la descomposición SVD
    inicio = time.time()
    try:
        beta_svd = regresion_SVD(X, y)
        tiempos["SVD"] = time.time() - inicio
        print("Coeficientes obtenidos (SVD):", beta_svd)
    except ValueError as e:
        tiempos["SVD"] = None
        print("Error en descomposición SVD:", e)

    # Comparación gráfica de tiempos de ejecución
    mostrar_grafico_tiempos(tiempos)

def mostrar_grafico_tiempos(tiempos):
    metodos = [metodo for metodo, tiempo in tiempos.items() if tiempo is not None]
    tiempos_filtrados = [tiempo for tiempo in tiempos.values() if tiempo is not None]

    plt.figure(figsize=(10, 6))
    plt.bar(metodos, tiempos_filtrados, color='skyblue')
    plt.xlabel("Método de Descomposición")
    plt.ylabel("Tiempo de Ejecución (segundos)")
    plt.title("Comparación de Tiempos de Ejecución de Métodos de Regresión")
    plt.show()
