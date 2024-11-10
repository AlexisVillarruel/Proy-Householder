import numpy as np

def validar_matriz_diseno(X):
    """Valida que la matriz X tenga al menos tantas filas como columnas y que tenga rango completo."""
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
    """Calcula la descomposición en valores singulares (SVD) de una matriz X usando una aproximación iterativa."""
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
    Sigma_pinv = np.linalg.pinv(Sigma)  # Usar pseudoinversa para estabilidad numérica
    beta = VT @ Sigma_pinv @ U.T @ y
    return beta

def get_matrix_from_user(rows, cols, matrix_name="X"):
    """Solicita al usuario que ingrese los elementos de una matriz"""
    A = np.zeros((rows, cols))
    print(f"Ingrese los elementos de la matriz {matrix_name} ({rows}x{cols}):")

    for i in range(rows):
        for j in range(cols):
            A[i, j] = float(input(f"Elemento [{i+1},{j+1}]: "))

    return A

def main():
    # Paso 1: Definir las dimensiones de la matriz de diseño X
    n = int(input("Ingrese el número de muestras (número de filas de X): "))
    p = int(input("Ingrese el número de características (número de columnas de X): "))

    # Paso 2: Ingresar la matriz X
    X = get_matrix_from_user(n, p, "X")

    # Paso 3: Ingresar el vector de respuesta y
    print("\nIngrese los elementos del vector y:")
    y = np.zeros(n)
    for i in range(n):
        y[i] = float(input(f"Elemento y[{i+1}]: "))

    # Mostrar la matriz X y el vector y ingresados
    print("\nMatriz X ingresada:")
    print(X)
    print("\nVector y ingresado:")
    print(y)

    # Paso 4: Intentar resolver el problema de mínimos cuadrados
    try:
        beta_hat = regresion_SVD(X, y)
        print("\nCoeficientes estimados (beta):")
        print(beta_hat)
    except ValueError as e:
        print("\nError en los datos de entrada:", e)
    except np.linalg.LinAlgError as e:
        print("\nError en el cálculo de beta (problema con SVD):", e)

if __name__ == "__main__":
    main()
