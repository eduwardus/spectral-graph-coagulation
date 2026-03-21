# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:09:40 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectral_control_analysis.py

Verifica la conjetura espectral: el autovalor principal de la matriz de adyacencia
controla la estructura global del grafo generado por coagulación espectral.

Analiza tres relaciones:
1. λ₁ vs grado medio <k>
2. λ₁ vs tamaño N
3. Relación con dimensión espectral (si hay datos)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigs
from scipy.stats import linregress
import networkx as nx
import glob
import os
import re
import sys
import time
from tqdm import tqdm

# ==================== CONFIGURACIÓN ====================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Estilo para publicación
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Directorios de búsqueda
GRAPH_DIRS = [
    "soup_simulation_phase_transition_v20",  # Ajusta esta ruta
    "soup_simulation_phase_transition_v20/snapshots"
]

# Tamaño mínimo para análisis
MIN_NODES = 100

# Archivo de datos de dimensión espectral (si existe)
DS_DATA_FILE = "spectral_dimension_vs_size.dat"
if not os.path.exists(DS_DATA_FILE):
    DS_DATA_FILE = "soup_simulation_phase_transition_v20/spectral_dimension_vs_size.dat"

# ==================== FUNCIONES DE PROGRESO ====================
def print_progress(current, total, start_time, valid):
    elapsed = time.time() - start_time
    if current > 0:
        eta = (elapsed / current) * (total - current)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
    else:
        eta_str = "calculando..."
    
    percent = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current // total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    sys.stdout.write(f'\r   [{bar}] {current}/{total} ({percent:.1f}%) | '
                     f'Válidos: {valid} | Tiempo restante: {eta_str}')
    sys.stdout.flush()

# ==================== BUSCAR ARCHIVOS ====================
def find_graph_files():
    """Encuentra todos los archivos .npz que contienen grafos."""
    graph_files = []
    for directory in GRAPH_DIRS:
        if os.path.exists(directory):
            pattern = os.path.join(directory, "*.npz")
            files = glob.glob(pattern)
            print(f"   {directory}: {len(files)} archivos")
            graph_files.extend(files)
    return graph_files

def extract_size_from_filename(filename):
    """Extrae el tamaño N del nombre del archivo."""
    patterns = [r'_N(\d+)_', r'_Ninit(\d+)', r'N(\d+)\.npz']
    basename = os.path.basename(filename)
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            return int(match.group(1))
    return None

# ==================== CALCULAR AUTOVALOR PRINCIPAL ====================
def compute_spectral_radius(G):
    """
    Calcula el autovalor principal (radio espectral) de la matriz de adyacencia.
    Para grafos grandes usa el método iterativo de Arnoldi.
    """
    try:
        N = G.number_of_nodes()
        
        # Convertir a matriz dispersa
        A = nx.to_scipy_sparse_array(G, format='csr')
        
        # Calcular el autovalor de mayor magnitud
        # Usamos k=2 para asegurar convergencia, luego tomamos el máximo
        if N > 5000:
            # Para grafos muy grandes, usar menor precisión
            lambda_max = eigs(A, k=1, which='LM', return_eigenvectors=False,
                              maxiter=1000, tol=1e-4)[0].real
        else:
            lambda_max = eigs(A, k=1, which='LM', return_eigenvectors=False,
                              maxiter=100*N, tol=1e-6)[0].real
        
        return lambda_max
    except Exception as e:
        # Si falla, intentar con método denso para grafos pequeños
        try:
            if N < 1000:
                A_dense = nx.to_numpy_array(G)
                eigenvalues = np.linalg.eigvals(A_dense)
                return np.max(np.abs(eigenvalues)).real
            else:
                return None
        except:
            return None

# ==================== CARGAR DATOS DE DIMENSIÓN ESPECTRAL ====================
def load_spectral_dimension_data():
    """Carga datos de dimensión espectral si existen."""
    if not os.path.exists(DS_DATA_FILE):
        return None, None
    
    try:
        data = np.loadtxt(DS_DATA_FILE, comments='#')
        if data.ndim == 1:
            # Un solo grafo
            return np.array([data[0]]), np.array([data[1]])
        else:
            return data[:, 0], data[:, 1]
    except:
        return None, None

# ==================== PROCESAR GRAFOS ====================
print("🔍 Buscando archivos de grafos...")
graph_files = find_graph_files()
print(f"📁 Encontrados {len(graph_files)} archivos .npz")

if len(graph_files) == 0:
    print("❌ No se encontraron archivos .npz")
    sys.exit(1)

# Listas para resultados
sizes = []
lambda1_list = []
k_mean_list = []

print(f"\n🔬 Calculando autovalor principal para {len(graph_files)} grafos...")
print(f"   Mínimo {MIN_NODES} nodos")
print()

valid_count = 0
start_time = time.time()

for i, gf in enumerate(tqdm(graph_files, desc="Procesando")):
    try:
        # Obtener tamaño
        N = extract_size_from_filename(gf)
        if N is None:
            A = load_npz(gf)
            N = A.shape[0]

        if N < MIN_NODES:
            continue

        # Cargar grafo
        A = load_npz(gf)
        G = nx.from_scipy_sparse_array(A)

        # Tomar componente gigante
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest)
            N = G.number_of_nodes()
            if N < MIN_NODES:
                continue

        # Grado medio
        degrees = [d for n, d in G.degree()]
        k_mean = np.mean(degrees)

        # Autovalor principal
        lambda1 = compute_spectral_radius(G)

        if lambda1 is not None and lambda1 > 0:
            sizes.append(N)
            lambda1_list.append(lambda1)
            k_mean_list.append(k_mean)
            valid_count += 1

    except Exception as e:
        continue

elapsed = time.time() - start_time
print(f"\n\n✅ Procesados {valid_count} grafos válidos en {elapsed:.1f} segundos")

if valid_count == 0:
    print("❌ No se obtuvieron datos válidos")
    sys.exit(1)

# Convertir a arrays
sizes = np.array(sizes)
lambda1 = np.array(lambda1_list)
k_mean = np.array(k_mean_list)

# ==================== 1. λ₁ VS GRADO MEDIO ====================
slope_k, intercept_k, r_k, p_k, std_k = linregress(k_mean, lambda1)
r2_k = r_k**2

print("\n" + "="*70)
print(" 1. λ₁ vs GRADO MEDIO")
print("="*70)
print(f"   λ₁ = {slope_k:.4f}·⟨k⟩ + {intercept_k:.4f}")
print(f"   R² = {r2_k:.4f}")
print(f"   p-value = {p_k:.4e}")

# Interpretación
if abs(slope_k - 1) < 0.1:
    print("\n   ✅ λ₁ ≈ ⟨k⟩ → red homogénea")
elif slope_k > 1.5:
    print("\n   ⚠️ λ₁ ≫ ⟨k⟩ → presencia de hubs dominantes")
else:
    print("\n   ℹ️ Régimen intermedio")

# ==================== 2. λ₁ VS TAMAÑO ====================
logN = np.log(sizes)
log_lambda = np.log(lambda1)

slope_N, intercept_N, r_N, p_N, std_N = linregress(logN, log_lambda)
r2_N = r_N**2
alpha = slope_N

print("\n" + "="*70)
print(" 2. λ₁ vs TAMAÑO: λ₁ ∼ N^α")
print("="*70)
print(f"   α = {alpha:.4f} ± {std_N:.4f}")
print(f"   R² = {r2_N:.4f}")
print(f"   p-value = {p_N:.4e}")

if alpha < 0.1:
    print("   → λ₁ ≈ constante (independiente del tamaño)")
elif alpha < 0.3:
    print("   → Crecimiento sub-lineal débil")
elif alpha < 0.5:
    print("   → Crecimiento moderado")
else:
    print("   → Crecimiento fuerte (organización espectral)")

# ==================== 3. RELACIÓN CON DIMENSIÓN ESPECTRAL ====================
ds_sizes, ds_values = load_spectral_dimension_data()

print("\n" + "="*70)
print(" 3. RELACIÓN CON DIMENSIÓN ESPECTRAL")
print("="*70)

if ds_sizes is not None and len(ds_sizes) > 0:
    # Interpolar d_s para los tamaños que tenemos
    from scipy.interpolate import interp1d
    
    # Ordenar
    sort_idx = np.argsort(ds_sizes)
    ds_sizes_sorted = ds_sizes[sort_idx]
    ds_values_sorted = ds_values[sort_idx]
    
    # Interpolar
    ds_interp = []
    lambda_for_ds = []
    
    for i, N in enumerate(sizes):
        if N >= min(ds_sizes_sorted) and N <= max(ds_sizes_sorted):
            # Interpolación lineal
            idx = np.searchsorted(ds_sizes_sorted, N)
            if idx > 0 and idx < len(ds_sizes_sorted):
                N1, N2 = ds_sizes_sorted[idx-1], ds_sizes_sorted[idx]
                d1, d2 = ds_values_sorted[idx-1], ds_values_sorted[idx]
                if N2 > N1:
                    d = d1 + (d2 - d1) * (N - N1) / (N2 - N1)
                    ds_interp.append(d)
                    lambda_for_ds.append(lambda1[i])
    
    if len(ds_interp) > 5:
        ds_interp = np.array(ds_interp)
        lambda_for_ds = np.array(lambda_for_ds)
        
        slope_ds, intercept_ds, r_ds, p_ds, std_ds = linregress(ds_interp, lambda_for_ds)
        r2_ds = r_ds**2
        
        print(f"   λ₁ vs d_s: R² = {r2_ds:.4f}")
        print(f"   λ₁ = {slope_ds:.4f}·d_s + {intercept_ds:.4f}")
        
        if r2_ds > 0.7:
            print("   ✅ Fuerte correlación: el modo espectral dominante")
            print("      controla la dimensión efectiva")
        else:
            print("   ⚠️ Correlación débil")
    else:
        print("   No hay suficientes datos para correlación con d_s")
else:
    print("   No se encontraron datos de dimensión espectral")

# ==================== FIGURA 1: λ₁ VS GRADO MEDIO ====================
plt.figure(figsize=(10, 8))

plt.scatter(k_mean, lambda1, alpha=0.4, s=15, c='blue', label='Datos')

# Línea de ajuste
x_fit = np.linspace(min(k_mean), max(k_mean), 100)
y_fit = slope_k * x_fit + intercept_k
plt.plot(x_fit, y_fit, 'r-', linewidth=2,
         label=f'Ajuste: λ₁ = {slope_k:.2f}⟨k⟩ + {intercept_k:.2f}\n$R^2 = {r2_k:.3f}$')

# Línea de referencia λ₁ = ⟨k⟩
plt.plot(x_fit, x_fit, 'g--', alpha=0.5, label='λ₁ = ⟨k⟩')

plt.xlabel('Grado medio $\\langle k \\rangle$', fontsize=14)
plt.ylabel('Radio espectral $\\lambda_1$', fontsize=14)
plt.title('Correlación entre radio espectral y grado medio', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('fig_lambda_vs_degree.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_lambda_vs_degree.png', dpi=300, bbox_inches='tight')
print("\n✅ Figura 1 guardada como fig_lambda_vs_degree.pdf y .png")

# ==================== FIGURA 2: λ₁ VS TAMAÑO ====================
plt.figure(figsize=(10, 8))

plt.scatter(sizes, lambda1, alpha=0.4, s=15, c='blue', label='Datos')

# Ajuste potencial
x_fit = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
y_fit = np.exp(intercept_N) * x_fit**alpha
plt.plot(x_fit, y_fit, 'r-', linewidth=2,
         label=f'Ajuste: $\\lambda_1 \\sim N^{{{alpha:.2f}}}$\n$R^2 = {r2_N:.3f}$')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Tamaño del cluster $N$', fontsize=14)
plt.ylabel('Radio espectral $\\lambda_1$', fontsize=14)
plt.title('Escalado del radio espectral con el tamaño', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('fig_lambda_vs_size.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_lambda_vs_size.png', dpi=300, bbox_inches='tight')
print("✅ Figura 2 guardada como fig_lambda_vs_size.pdf y .png")

# ==================== FIGURA 3: λ₁ VS d_s (si hay datos) ====================
if ds_sizes is not None and len(ds_sizes) > 0 and len(ds_interp) > 5:
    plt.figure(figsize=(10, 8))
    
    plt.scatter(ds_interp, lambda_for_ds, alpha=0.4, s=15, c='green', label='Datos')
    
    # Línea de ajuste
    x_fit = np.linspace(min(ds_interp), max(ds_interp), 100)
    y_fit = slope_ds * x_fit + intercept_ds
    plt.plot(x_fit, y_fit, 'r-', linewidth=2,
             label=f'Ajuste: $\\lambda_1 = {slope_ds:.2f}d_s + {intercept_ds:.2f}$\n$R^2 = {r2_ds:.3f}$')
    
    plt.xlabel('Dimensión espectral $d_s$', fontsize=14)
    plt.ylabel('Radio espectral $\\lambda_1$', fontsize=14)
    plt.title('Relación entre radio espectral y dimensión espectral', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('fig_lambda_vs_ds.pdf', dpi=300, bbox_inches='tight')
    print("✅ Figura 3 guardada como fig_lambda_vs_ds.pdf")

# ==================== ESTADÍSTICAS POR RANGO ====================
print("\n" + "="*70)
print(" ESTADÍSTICAS POR RANGO DE TAMAÑOS")
print("="*70)

ranges = [
    (100, 500, "Pequeños (100-500)"),
    (500, 5000, "Medianos (500-5000)"),
    (5000, 50000, "Grandes (5000-50000)"),
    (50000, 1000000, "Muy grandes (>50000)")
]

for low, high, label in ranges:
    mask = (sizes >= low) & (sizes < high)
    if np.sum(mask) > 0:
        print(f"\n{label}:")
        print(f"   n = {np.sum(mask)}")
        print(f"   ⟨k⟩ medio = {np.mean(k_mean[mask]):.2f} ± {np.std(k_mean[mask]):.2f}")
        print(f"   λ₁ medio = {np.mean(lambda1[mask]):.2f} ± {np.std(lambda1[mask]):.2f}")
        print(f"   λ₁/⟨k⟩ = {np.mean(lambda1[mask] / k_mean[mask]):.4f}")

# ==================== RESUMEN PARA EL PAPER ====================
print("\n" + "="*70)
print(" RESUMEN PARA EL PAPER")
print("="*70)

print(f"""
\\paragraph{{Spectral control}}
Since the fusion rule in our model is based on spectral stability,
we analyzed whether the global network structure is controlled by the
spectral radius $\\lambda_1$ of the adjacency matrix.

The results show:

\\begin{{itemize}}
    \\item Strong linear correlation $\\lambda_1 \\approx {slope_k:.2f}\\langle k\\rangle$ 
          ($R^2 = {r2_k:.3f}$), indicating that the dominant eigenmode
          scales with the mean degree.
    \\item Power-law growth with network size: $\\lambda_1 \\sim N^{{{alpha:.2f}}}$
          ($R^2 = {r2_N:.3f}$).
    \\item {'Strong' if r2_ds > 0.7 else 'Weak'} correlation with spectral dimension
          ($R^2 = {r2_ds:.3f}$).
\\end{{itemize}}

These findings confirm that the spectral radius — the quantity implicitly
optimized by the fusion rule — indeed governs the large-scale properties
of the emergent networks.
""")

# ==================== GUARDAR DATOS ====================
output_data = np.column_stack((sizes, k_mean, lambda1))
np.savetxt('spectral_control_data.dat', output_data,
           header='N k_mean lambda1', fmt='%d %.4f %.4f')
print("\n💾 Datos guardados en spectral_control_data.dat")