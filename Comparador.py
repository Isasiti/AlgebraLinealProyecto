import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
import matplotlib.pyplot as plt

# ---------- Funciones base ----------
def crear_malla(n):
    """Crea nodos y barras en una malla n x n"""
    nodos = [(i, j) for j in range(n) for i in range(n)]
    barras = []
    for j in range(n):
        for i in range(n):
            idx = i + j * n
            if i < n - 1:  # barra horizontal
                barras.append((idx, idx + 1))
            if j < n - 1:  # barra vertical
                barras.append((idx, idx + n))
    return nodos, barras

def propiedades_barra(nodos, i, j):
    xi, yi = nodos[i]
    xj, yj = nodos[j]
    L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
    c = (xj - xi) / L
    s = (yj - yi) / L
    return L, c, s

def matriz_rigidez_local(E, A, L, c, s):
    k = (E * A / L) * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])
    return k

def ensamblar_sistema(n):
    nodos, barras = crear_malla(n)
    num_nodos = len(nodos)
    num_gdl = num_nodos * 2
    K = np.zeros((num_gdl, num_gdl))
    F = np.zeros(num_gdl)
    
    E = 200e9
    A = 0.01

    for ni, nj in barras:
        L, c, s = propiedades_barra(nodos, ni, nj)
        k_local = matriz_rigidez_local(E, A, L, c, s)
        indices = [2*ni, 2*ni+1, 2*nj, 2*nj+1]
        for i in range(4):
            for j in range(4):
                K[indices[i], indices[j]] += k_local[i, j]

    # Fuerza descendente en nodo superior derecho
    F[-1] = -1000

    # Restricción en nodo (0, 0): GDL 0 y 1
    restricciones = [0, 1]
    K_mod = np.copy(K)
    F_mod = np.copy(F)
    for r in restricciones:
        K_mod[r, :] = 0
        K_mod[:, r] = 0
        K_mod[r, r] = 1
        F_mod[r] = 0

    return K_mod, F_mod

# ---------- Métodos ----------
def resolver_gauss(K, F):
    return np.linalg.solve(K, F)

def resolver_LU(K, F):
    P, L, U = scipy.linalg.lu(K)
    y = scipy.linalg.solve(L, P @ F)
    return scipy.linalg.solve(U, y)

def resolver_cholesky(K, F):
    L = np.linalg.cholesky(K)
    y = np.linalg.solve(L, F)
    return np.linalg.solve(L.T, y)

def resolver_gradiente_conjugado(K, F):
    x, info = scipy.sparse.linalg.cg(K, F)
    return x

# ---------- Comparación escalable ----------
def comparar_estructuras(max_n=6):
    metodos = {
        "Gauss": resolver_gauss,
        "LU": resolver_LU,
        "Cholesky": resolver_cholesky,
        "Gradiente Conjugado": resolver_gradiente_conjugado
    }

    tiempos = {m: [] for m in metodos}
    tamaños = []

    for n in range(2, max_n + 1):
        print(f"\nEvaluando malla {n}x{n}...")
        K, F = ensamblar_sistema(n)
        tamaños.append(len(F))  # número de GDL

        for nombre, metodo in metodos.items():
            try:
                inicio = time.time()
                metodo(K, F)
                duracion = time.time() - inicio
            except Exception as e:
                duracion = None
            tiempos[nombre].append(duracion)

    return tamaños, tiempos

# ---------- Visualización ----------
def graficar(tamaños, tiempos):
    for nombre, t in tiempos.items():
        plt.plot(tamaños, t, label=nombre)
    plt.xlabel("Tamaño del sistema (GDL)")
    plt.ylabel("Tiempo (s)")
    plt.title("Comparación de métodos de resolución")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------- Ejecutar todo ----------
tamaños, tiempos = comparar_estructuras(max_n=10)
graficar(tamaños, tiempos)
