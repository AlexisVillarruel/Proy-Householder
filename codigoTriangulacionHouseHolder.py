import numpy as np

def householder_reflection(A, tol=1e-10):
    """
    Realiza la descomposición QR usando reflexiones de Householder.

    Parámetros:
    - A (numpy array): Matriz a descomponer de tamaño (m, n), donde m >= n.
    - tol (float): Umbral de tolerancia para considerar valores como cero en R.

    Retorna:
    - Q (numpy array): Matriz ortogonal de tamaño (m, m).
    - R (numpy array): Matriz triangular superior de tamaño (m, n) con valores menores al umbral en la parte inferior establecidos a cero.
    - str: Mensaje de error si las condiciones no se cumplen.
    """
    m, n = A.shape
    if m < n:
        return None, None, "Error: La matriz debe tener m >= n para realizar la descomposición QR."
    
    Q = np.eye(m)  # Matriz identidad de tamaño (m, m) para iniciar Q
    R = A.copy()  # Copia de A para R, que se modificará en el proceso

    for i in range(n):  # Itera sobre cada columna de A
        x = R[i:, i]  # Subvector desde la posición i hacia abajo
        norm_x = np.linalg.norm(x)  # Calcula la norma del subvector
        if norm_x == 0:
            continue  # Si la norma es cero, no se requiere transformación
        
        # Vector de reflexión v
        e1 = np.zeros_like(x)
        e1[0] = norm_x
        v = x - e1
        v = v / np.linalg.norm(v)  # Normaliza v

        # Matriz de reflexión H_i
        H_i = np.eye(m)
        H_i[i:, i:] -= 2.0 * np.outer(v, v)  # Aplica reflexión en submatriz
        
        # Actualiza R y acumula la transformación en Q
        R = H_i @ R
        Q = Q @ H_i

    # Forzar a cero los elementos de R que están debajo de la diagonal y son menores que tol
    R = np.triu(R, k=0) * (np.abs(R) > tol)

    # Verificación de que Q es ortogonal
    if not np.allclose(Q.T @ Q, np.eye(m), atol=tol):
        return None, None, "Error: La matriz Q no es ortogonal."

    return Q, R, None

def linear_regression_householder(X, y, tol=1e-10):
    """
    Calcula los coeficientes beta en un modelo de regresión lineal mediante QR por Householder.
    
    Parámetros:
    - X (numpy array): Matriz de diseño de tamaño (m, n).
    - y (numpy array): Vector de observaciones de tamaño (m,).
    - tol (float): Umbral de tolerancia para considerar valores en R como cero.
    
    Retorna:
    - beta_hat (numpy array): Estimación de los coeficientes beta.
    - str: Mensaje de error si las condiciones no se cumplen.
    """
    m, n = X.shape
    # Verificar que X tiene rango completo
    if np.linalg.matrix_rank(X) < n:
        return None, "Error: Las columnas de X deben ser linealmente independientes."
    
    Q, R, error = householder_reflection(X, tol)
    if error:
        return None, error

    # Mostrar las matrices Q y R
    print("Matriz Q:")
    print(Q)
    print("\nMatriz R:")
    print(R)

    # Multiplicación de Q^T * y
    Qt_y = Q.T @ y

    # Verificar que R esté bien condicionada para estabilidad numérica
    if np.linalg.cond(R[:n, :]) > 1e10:
        return None, "Error: La matriz R es mal condicionada o singular."

    # Resolver R * beta = Q^T * y
    try:
        beta_hat = np.linalg.solve(R[:n, :], Qt_y[:n])
        print("\nEstimación de beta_hat:", beta_hat)
    except np.linalg.LinAlgError:
        return None, "Error: No se pudo resolver el sistema lineal."

    return beta_hat, None

# --- Inputs (definir X y y aquí) ---
# Ejemplo de matriz X y vector y que cumple todas las condiciones
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 10],
    [10, 11, 13]
])  # Matriz de diseño (m, n) donde m > n y rango completo

y = np.array([1, 2, 3, 4])  # Vector de observaciones (m,)

# Realiza la regresión lineal usando Householder QR
beta_hat, error = linear_regression_householder(X, y)
if error:
    print("Error:", error)
else:
    print("\nEstimación de beta_hat:", beta_hat)
