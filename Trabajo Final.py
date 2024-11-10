import numpy as np 
def cholesky_decomposition(A):
    """Realiza la descomposición de Cholesky de una matriz A"""
    n = A.shape[0]
    L = np.zeros((n, n))

    # Verificar que A es simétrica
    if not np.allclose(A, A.T):
        raise ValueError("La matriz no es simétrica.")

    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:  # Elementos diagonales
                val = A[i][i] - sum_k
                if val <= 0:
                    raise ValueError("La matriz no es definida positiva.")
                L[i][j] = np.sqrt(val)
            else:
                L[i][j] = (A[i][j] - sum_k) / L[j][j]

    return L

def solve_cholesky(X, y):
    """Resuelve la regresión lineal usando Cholesky."""
    XtX = X.T @ X
    Xty = X.T @ y
    L = cholesky_decomposition(XtX)

    # Resolver Lz = X^T y
    z = np.linalg.solve(L, Xty)

    # Resolver L.T * beta = z
    beta_hat = np.linalg.solve(L.T, z)

    return beta_hat

def householder_reflection(A, tol=1e-10):
    """Descomposición QR por reflexiones de Householder."""
    m, n = A.shape
    if m < n:
        return None, None, "Error: m debe ser >= n."
    
    Q = np.eye(m)  
    R = A.copy()  

    for i in range(n):  # Itera sobre cada columna de A
        x = R[i:, i]  
        norm_x = np.linalg.norm(x)  # Calcula la norma del subvector
        if norm_x == 0:
            continue  # Si la norma es cero, no se requiere transformación
        
        e1 = np.zeros_like(x)
        e1[0] = norm_x
        v = x - e1
        v = v / np.linalg.norm(v)  # Normaliza v

        H_i = np.eye(m)
        H_i[i:, i:] -= 2.0 * np.outer(v, v) 
        
        R = H_i @ R
        Q = Q @ H_i

    R = np.triu(R, k=0) * (np.abs(R) > tol)

    # Verificación de que Q es ortogonal
    if not np.allclose(Q.T @ Q, np.eye(m), atol=tol):
        return None, None, "Error: La matriz Q no es ortogonal."

    return Q, R, None

def linear_regression_householder(X, y, tol=1e-10):
    """Calcula beta usando QR de Householder."""
    m, n = X.shape
    # Verificar que X tiene rango completo
    if np.linalg.matrix_rank(X) < n:
        return None, "Error: Las columnas de X deben ser linealmente independientes."
    
    Q, R, error = householder_reflection(X, tol)
    if error:
        return None, error

    print("Matriz Q:")
    print(Q)
    print("\nMatriz R:")
    print(R)

    Qt_y = Q.T @ y

    # Verificar que R esté bien condicionada para estabilidad numérica
    if np.linalg.cond(R[:n, :]) > 1e10:
        return None, "Error: La matriz R es mal condicionada o singular."

    try:
        beta_hat = np.linalg.solve(R[:n, :], Qt_y[:n])
        print("\nEstimación de beta_hat:", beta_hat)
    except np.linalg.LinAlgError:
        return None, "Error: No se pudo resolver el sistema lineal."

    return beta_hat, None

  
def gram_schmidt_linear_regression(X, y):
    # Validaciones de entrada
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X e y deben ser matrices y vectores de tipo numpy.ndarray.")
        
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X debe ser una matriz de 2 dimensiones y y un vector de 1 dimensión.")
        
    n, p = X.shape
    if n < p:
        raise ValueError("El número de muestras (n) debe ser mayor o igual al número de características (p).")
        
    if len(y) != n:
        raise ValueError("La longitud de y debe coincidir con el número de filas de X.")

    # Inicializar matrices Q y R
    Q = np.zeros((n, p))
    R = np.zeros((p, p))
    
    # Ortogonalización de Gram-Schmidt
    for j in range(p):
        v = X[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], X[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        
        # Verificar que no se genere un vector nulo, lo cual indicaría dependencia lineal
        if R[j, j] == 0:
            raise ValueError("Las columnas de X deben ser linealmente independientes.")
            
        Q[:, j] = v / R[j, j]
    beta_hat = np.linalg.solve(R, Q.T @ y)
    
    return beta_hat

def validar_matriz_diseno(X):
    n, p = X.shape

    # Verificar que X tenga más filas que columnas
    if n < p:
        raise ValueError("La matriz de diseño X debe tener al menos tantas filas como columnas (n >= p).")

    # Verificar que el rango de X sea completo
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("La matriz de diseño X no tiene rango completo (colinealidad detectada).")

def validar_vector_respuesta(X, y):
    """Valida que el vector de respuesta y tenga el mismo número de filas que X."""
    if X.shape[0] != y.size:
        raise ValueError("El vector de respuesta y debe tener el mismo número de filas que X.")

def svd(X, tol=1e-10, max_iter=100):
    """Realiza la descomposición SVD de una matriz X."""
    n, m = X.shape
    U = np.eye(n)  # Matriz ortogonal U
    V = np.eye(m)  # Matriz ortogonal V
    A = X.copy()   # Copia de X

    for _ in range(max_iter):
        for i in range(m):
            for j in range(i + 1, m):
                a, b = A[:, i], A[:, j]
                theta = 0.5 * np.arctan2(2 * np.dot(a, b), np.dot(a, a) - np.dot(b, b))

                # Matriz de rotación
                c, s = np.cos(theta), np.sin(theta)
                A[:, [i, j]] = A[:, [i, j]] @ np.array([[c, s], [-s, c]])
                V[:, [i, j]] = V[:, [i, j]] @ np.array([[c, s], [-s, c]])

        if np.allclose(np.triu(A, 1), 0, atol=tol):
            break

    Sigma = np.sqrt(np.sum(A**2, axis=0))
    U = X @ V / Sigma.clip(min=tol)  # Clip para evitar división por cero

    return U, np.diag(Sigma), V.T

def regresion_SVD(X, y):
    """Calcula los coeficientes de regresión de mínimos cuadrados usando la descomposición SVD."""
    validar_matriz_diseno(X)
    validar_vector_respuesta(X, y)

    U, Sigma, VT = svd(X)
    Sigma_pinv = np.linalg.pinv(Sigma)  
    beta = VT @ Sigma_pinv @ U.T @ y
    return beta

def menu_creacion_matriz():
    print("Opciones para la matriz y el vector:")
    print("1. Crear matriz y vector manualmente")
    print("2. Generar matriz y vector aleatoriamente")
    data_choice = int(input("Ingrese el número de la opción deseada: "))

    if data_choice == 1:
        rows = int(input("Ingrese el número de filas para la matriz: "))
        cols = int(input("Ingrese el número de columnas para la matriz: "))
        X = np.zeros((rows, cols))
        
        print("Ingrese los elementos de la matriz:")
        for i in range(rows):
            for j in range(cols):
                X[i, j] = float(input(f"Elemento [{i+1},{j+1}]: "))
        
        y = np.zeros(rows)
        print("Ingrese los elementos del vector:")
        for i in range(rows):
            y[i] = float(input(f"Elemento [{i+1}]: "))
    else:
        rows = int(input("Ingrese el número de filas para la matriz: "))
        cols = int(input("Ingrese el número de columnas para la matriz: "))
        X = np.random.rand(rows, cols)
        y = np.random.rand(rows)
        
        print("\nMatriz generada aleatoriamente:")
        print(X)
        print("\nVector generado aleatoriamente:")
        print(y)

    return X, y

def menu_descomposicion():
    print("\nSeleccione una opción para la descomposición:")
    print("1. Cholesky")
    print("2. Householder")
    print("3. Gram-Schmidt")
    print("4. SVD")
    choice = int(input("Ingrese el número de la opción deseada: "))
    return choice

def main():
    X, y = menu_creacion_matriz()
    choice = menu_descomposicion()
    
    if choice == 1:
        try:
            beta = solve_cholesky(X, y)
            print("Coeficientes obtenidos (Cholesky):", beta)
        except ValueError as e:
            print("Error:", e)
    elif choice == 2:
        beta, error = linear_regression_householder(X, y)
        if error:
            print("Error:", error)
        else:
            print("Coeficientes obtenidos (Householder):", beta)
    elif choice == 3:
        try:
            beta = gram_schmidt_linear_regression(X, y)
            print("Coeficientes obtenidos (Gram-Schmidt):", beta)
        except ValueError as e:
            print("Error:", e)
    elif choice == 4:
        try:
            beta = regresion_SVD(X, y)
            print("Coeficientes obtenidos (SVD):", beta)
        except ValueError as e:
            print("Error:", e)
    else:
        print("Opción no válida.")

main()
