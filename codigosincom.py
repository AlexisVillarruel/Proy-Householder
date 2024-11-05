import numpy as np
import matplotlib.pyplot as plt
import time

def householder_reflection(A):
    m, n = A.shape
    if m < n:
        return None, None, "Error: La matriz debe tener m >= n para realizar la descomposición QR."

    Q = np.eye(m)
    R = A.copy()

    for i in range(n):
        x = R[i:, i]
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            continue
        e1 = np.zeros_like(x)
        e1[0] = norm_x
        v = x - e1
        v = v / np.linalg.norm(v)
        
        H_i = np.eye(m)
        H_i[i:, i:] -= 2.0 * np.outer(v, v)
        R = H_i @ R
        Q = Q @ H_i

    if R.shape[0] != R.shape[1]:
        return None, None, "Error: La matriz R no es cuadrada."
    if not np.allclose(Q.T @ Q, np.eye(m), atol=1e-10):
        return None, None, "Error: Q no es ortogonal."

    return Q, R, None

def linear_regression_householder(X, y):
    m, n = X.shape
    if np.linalg.matrix_rank(X) < n:
        return None, "Error: Las columnas de X deben ser linealmente independientes."

    Q, R, error = householder_reflection(X)
    if error:
        return None, error

    Qt_y = Q.T @ y
    if np.linalg.cond(R[:n, :]) > 1e10:
        return None, "Error: La matriz R es mal condicionada."

    try:
        beta_hat = np.linalg.solve(R[:n, :], Qt_y[:n])
    except np.linalg.LinAlgError:
        return None, "Error: No se pudo resolver el sistema lineal."

    return beta_hat, None

# Medición de tiempos
matrix_sizes = [(100, 10), (200, 20), (300, 30), (400, 40), (500, 50)]
times = []

for m, n in matrix_sizes:
    A = np.random.rand(m, n)
    start_time = time.time()
    householder_reflection(A)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)

# Gráfico de tiempos
sizes = [m * n for m, n in matrix_sizes]

plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o', linestyle='-', color='b')
plt.xlabel("Tamaño de la Matriz (m x n)")
plt.ylabel("Tiempo de Ejecución (segundos)")
plt.title("Tiempo de Ejecución de Descomposición QR mediante Householder")
plt.grid(True)
plt.show()
