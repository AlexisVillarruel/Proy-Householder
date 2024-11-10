import numpy as np

def verificar_datos_reales(matriz, vector):
    """
    Input
    - Matriz X
    - vector y
    """
    
    if not np.isrealobj(matriz) or not np.isrealobj(vector):    # Condición: Verifica si matriz y vector contienen solo números reales
        return False, "Error: Los datos deben ser números reales. El programa aun no acepta números complejos."
    
    
    if not np.all(np.isfinite(matriz)) or not np.all(np.isfinite(vector)):  # Condición: Verifica que todos los valores sean finitos o no numericos
        return False, "Error: Los datos contienen valores no numéricos o infinitos."

    return True, None

def householder_reflection(A, tol=1e-10):
    """
    Input
    - A : Matriz a descomponer de tamaño.
    - tol : Umbral de tolerancia para considerar valores como cero en R para estabilidad del algoritmo.

    Retorna:
    - Q : Matriz ortogonal
    - R : Matriz triangular superior
    """
    m, n = A.shape
    
    
    if m < n:       # Condición: Verifica que m >= n para que sea posible la descomposición QR
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


    R = np.triu(R, k=0) * (np.abs(R) > tol)

    if not np.allclose(Q.T @ Q, np.eye(m), atol=tol):   #Verifica que Q sea ortogonal
        return None, None, "Error: La matriz Q no es ortogonal."

    return Q, R, None

def linear_regression_householder(X, y, tol=1e-10):
    """
    Calcula los coeficientes beta en un modelo de regresión lineal mediante QR por Householder.

    Input:
    - X : Matriz de diseño de tamaño.
    - y : Vector de observaciones.

    Retorna:
    - beta_hat: Estimación de los coeficientes beta.
    """
    
    datos_reales, error = verificar_datos_reales(X, y)
    if not datos_reales:
        return None, error

    m, n = X.shape

    if np.linalg.matrix_rank(X) < n:        # Verifica que las columnas de X sean linealmente independientes
        return None, "Error: Las columnas de X deben ser linealmente independientes."
    
    Q, R, error = householder_reflection(X, tol)
    if error:
        return None, error

   
    print("Matriz Q:")
    print(Q)
    print("\nMatriz R:")
    print(R)

    Qt_y = Q.T @ y  

   
    if np.linalg.cond(R[:n, :]) > 1e10:          #Verifica que R esté bien condicionada para estabilidad numérica
        return None, "Error: La matriz R es mal condicionada o singular."

    try:
        beta_hat = np.linalg.solve(R[:n, :], Qt_y[:n])
        print("\nEstimación de beta_hat:", beta_hat)
    except np.linalg.LinAlgError:
        return None, "Error: No se pudo resolver el sistema lineal."

    return beta_hat, None

def menu():
    while True:
        print("\n--- Menú de Regresión Lineal usando QR por Householder ---")
        print("1. Ingresar matriz X")
        print("2. Ingresar vector y")
        print("3. Realizar regresión lineal")
        print("4. Salir")
        
        try:
            opcion = int(input("Seleccione una opción: "))
        except ValueError:
            print("Error: Ingrese un número válido.")
            continue
        
        if opcion == 1:
            try:
                filas = int(input("Ingrese el número de filas de X: "))
                columnas = int(input("Ingrese el número de columnas de X: "))
                print("Ingrese los elementos de la matriz X fila por fila:")
                X = np.array([list(map(float, input(f"Fila {i+1}: ").split())) for i in range(filas)])
            except ValueError:
                print("Error: Asegúrese de ingresar números reales.")
                continue
            
        elif opcion == 2:
            try:
                y = np.array(list(map(float, input("Ingrese los elementos del vector y separados por espacio: ").split())))
            except ValueError:
                print("Error: Asegúrese de ingresar números reales.")
                continue
            
        elif opcion == 3:
            if 'X' not in locals() or 'y' not in locals():
                print("Error: Debe ingresar la matriz X y el vector y antes de realizar la regresión.")
            else:
                beta_hat, error = linear_regression_householder(X, y)
                if error:
                    print("Error:", error)
                else:
                    print("\nEstimación de beta_hat:", beta_hat)
        
        elif opcion == 4:
            print("Saliendo del programa.")
            break
        
        else:
            print("Error: Seleccione una opción válida.")
menu()
