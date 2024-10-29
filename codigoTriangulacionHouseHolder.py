import numpy as np

def householder_reflection(A):
    """
    Realiza la descomposición QR usando reflexiones de Householder.
    
    Parámetros:
    A (numpy array): Matriz a descomponer de tamaño (m, n), donde m >= n.

    Retorna:
    Q (numpy array): Matriz ortogonal de tamaño (m, m).
    R (numpy array): Matriz triangular superior de tamaño (m, n).

    Complejidad: La complejidad principal de esta función es O(n^2 * m) debido al bucle y las operaciones de multiplicación de matrices.
    """
    # Verificar que la matriz tenga al menos tantas filas como columnas (m >= n).
    m, n = A.shape
    if m < n:
        raise ValueError("La matriz debe tener al menos tantas filas como columnas (m >= n).")

    # Inicializamos Q como una matriz identidad del tamaño de A. Complejidad O(m^2).
    Q = np.eye(m)
    # Copiamos A en R para no modificar los datos originales directamente. Complejidad O(m * n).
    R = A.copy()

    # Iteramos sobre cada columna de A. Complejidad del bucle: O(n) iteraciones.
    for i in range(n):
        # Seleccionamos el subvector x desde la posición i hacia abajo de la columna i de R. Complejidad O(m - i).
        x = R[i:, i]

        # Calculamos la norma del subvector x. Complejidad O(m - i).
        norm_x = np.linalg.norm(x)

        # Si la norma es cero, no hay necesidad de triangularizar esta columna. Complejidad O(1).
        if norm_x == 0:
            continue

        # Construimos el vector de referencia e1, con la misma longitud que x. Complejidad O(m - i).
        e1 = np.zeros_like(x)
        e1[0] = norm_x

        # Calculamos el vector de reflexión v. Complejidad O(m - i).
        v = x - e1
        v = v / np.linalg.norm(v)  # Normalizamos el vector v. Complejidad O(m - i).

        # Construimos la matriz de Householder H_i. Complejidad O((m - i)^2).
        H_i = np.eye(m)  # Empezamos con la matriz identidad. Complejidad O(m^2).
        H_i[i:, i:] -= 2.0 * np.outer(v, v)  # Modificamos la submatriz. Complejidad O((m - i)^2).

        # Multiplicamos H_i por R para triangularizar la columna i. Complejidad O(m * n).
        R = H_i @ R

        # Acumulamos la transformación en Q. Complejidad O(m^2).
        Q = Q @ H_i

    return Q, R

def linear_regression_householder(X, y):
    """
    Resuelve un problema de regresión lineal utilizando la descomposición QR con reflexiones de Householder.

    Parámetros:
    X (numpy array): Matriz de diseño de tamaño (m, n) (Debe cumplir m >= n y tener rango completo).
    y (numpy array): Vector de observaciones de tamaño (m,).

    Retorna:
    beta_hat (numpy array): Estimación de los coeficientes beta de tamaño (n,).

    Complejidad: La complejidad principal es O(n^2 * m) debido a la descomposición de Householder.
    """
    # Verificar dimensiones de X e y.
    m, n = X.shape
    if y.shape[0] != m:
        raise ValueError("El número de filas de X debe coincidir con la longitud de y.")
    
    # Verificar que las columnas de X sean linealmente independientes.
    if np.linalg.matrix_rank(X) < n:
        raise ValueError("Las columnas de X deben ser linealmente independientes (rango completo).")

    # Descomponer X en Q y R utilizando la descomposición de Householder.
    try:
        Q, R = householder_reflection(X)  # Complejidad O(n^2 * m)
    except Exception as e:
        raise ValueError(f"Error al calcular la descomposición QR: {str(e)}")

    # Multiplicamos Q^T por y para obtener el vector transformado Qt_y. Complejidad O(m^2).
    Qt_y = Q.T @ y

    # Verificamos que R sea cuadrada y no singular para resolver el sistema.
    if np.linalg.cond(R[:n, :]) > 1e10:  # Umbral para verificar si R es bien condicionada.
        raise ValueError("La matriz R es mal condicionada o singular. No se puede resolver el sistema.")

    # Resolver el sistema triangular superior R * beta_hat = Qt_y. Complejidad O(n^2).
    try:
        beta_hat = np.linalg.solve(R[:n, :], Qt_y[:n])
    except np.linalg.LinAlgError as e:
        raise ValueError(f"No se pudo resolver el sistema lineal: {str(e)}")

    return beta_hat

# Definir los inputs X y y para la regresión lineal.
# Matriz de diseño X (dimensión m x n).
X = np.array([[1, 1], [1, 2], [1, 3]])  # Ejemplo de matriz de diseño.

# Vector de observaciones y (dimensión m,).
y = np.array([1, 2, 3])  # Ejemplo de vector de observaciones.

# Ejecutar la función para calcular beta_hat con manejo de errores.
try:
    beta_hat = linear_regression_householder(X, y)
    print("Estimación de beta (beta_hat):", beta_hat)
except ValueError as error:
    print("Error:", error)
