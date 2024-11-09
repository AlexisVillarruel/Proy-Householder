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
    """Resuelve el problema de mínimos cuadrados usando la descomposición de Cholesky"""
    # Calculamos X^T * X y X^T * y
    XtX = X.T @ X
    Xty = X.T @ y

    # Descomposición de Cholesky de X^T * X
    L = cholesky_decomposition(XtX)

    # Resolver Lz = X^T y
    z = np.linalg.solve(L, Xty)

    # Resolver L.T * beta = z
    beta_hat = np.linalg.solve(L.T, z)

    return beta_hat

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
        beta_hat = solve_cholesky(X, y)
        print("\nCoeficientes estimados (beta):")
        print(beta_hat)
    except ValueError as e:
        print("\nError en el cálculo de beta:", e)

if __name__ == "__main__":
    main()
