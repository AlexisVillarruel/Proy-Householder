import numpy as np

def gram_schmidt_linear_regression(X, y):
    """
    Realiza la regresión lineal usando la ortogonalización de Gram-Schmidt para resolver el problema de mínimos cuadrados,
    con restricciones para asegurar que los datos de entrada sean adecuados.

    Parámetros:
    X : numpy.ndarray
        Matriz de diseño de tamaño n x p (n muestras, p características).
    y : numpy.ndarray
        Vector de salida de tamaño n.

    Retorna:
    beta_hat : numpy.ndarray
        Vector de coeficientes de tamaño p obtenido por mínimos cuadrados.

    Raises:
    ValueError: Si las dimensiones de X y y no son compatibles, si X tiene más columnas que filas, 
                o si alguna columna de X es linealmente dependiente de otras.
    """
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
    
    # Resolver el sistema de ecuaciones R * beta_hat = Q^T * y
    beta_hat = np.linalg.solve(R, Q.T @ y)
    
    return beta_hat