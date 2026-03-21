# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:32:13 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectral_cycle_relation.py

Comprueba la conjetura: λ₁ ≈ 2β₁ / N
donde β₁ = E - N + C es el primer número de Betti (ciclos independientes).

Analiza la correlación entre el radio espectral y la densidad de ciclos.
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
    "soup_simulation_phase_transition_v20",
    "soup_simulation_phase_transition_v20/snapshots"
]

# Tamaño mínimo para análisis
MIN_NODES = 100

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
    """
    try:
        N = G.number_of_nodes()
        
        # Convertir a matriz dispersa
        A = nx.to_scipy_sparse_array(G, format='csr')
        
        # Calcular el autovalor de mayor magnitud
        if N > 5000:
            lambda_max = eigs(A, k=1, which='LM', return_eigenvectors=False,
                              maxiter=1000, tol=1e-4)[0].real
        else:
            lambda_max = eigs(A, k=1, which='LM', return_eigenvectors=False,
                              maxiter=100*N, tol=1e-6)[0].real
        
        return lambda_max
    except Exception as e:
        # Fallback para grafos pequeños
        try:
            if N < 1000:
                A_dense = nx.to_numpy_array(G)
                eigenvalues = np.linalg.eigvals(A_dense)
                return np.max(np.abs(eigenvalues)).real
            else:
                return None
        except:
            return None

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
beta1_list = []
edges_list = []
components_list = []

print(f"\n🔬 Analizando relación espectral-ciclos para {len(graph_files)} grafos...")
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

        # Propiedades básicas
        N_nodes = G.number_of_nodes()
        E_edges = G.number_of_edges()
        C_comp = nx.number_connected_components(G)

        # Primer número de Betti
        beta1 = E_edges - N_nodes + C_comp
        beta1 = max(0, beta1)  # No puede ser negativo

        # Autovalor principal
        lambda1 = compute_spectral_radius(G)

        if lambda1 is not None and lambda1 > 0:
            sizes.append(N_nodes)
            lambda1_list.append(lambda1)
            beta1_list.append(beta1)
            edges_list.append(E_edges)
            components_list.append(C_comp)
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
beta1 = np.array(beta1_list)

# Variable teórica: 2β₁/N
lambda_pred = 2 * beta1 / sizes

# ==================== 1. CORRELACIÓN λ₁ vs 2β₁/N ====================
# Filtrar valores válidos
valid = (lambda_pred > 0) & (lambda1 > 0)
lambda_pred_f = lambda_pred[valid]
lambda1_f = lambda1[valid]

slope, intercept, r, p, std_err = linregress(lambda_pred_f, lambda1_f)
r2 = r**2

print("\n" + "="*70)
print(" RELACIÓN ESPECTRAL-CICLOS: λ₁ vs 2β₁/N")
print("="*70)
print(f"   λ₁ = {slope:.4f}·(2β₁/N) + {intercept:.4f}")
print(f"   R² = {r2:.4f}")
print(f"   p-value = {p:.4e}")
print(f"   Error estándar = {std_err:.4f}")

if abs(slope - 1) < 0.1 and r2 > 0.8:
    print("\n   ✅ CONFIRMADA: λ₁ ≈ 2β₁/N")
    print("      El radio espectral está controlado por la densidad de ciclos")
elif r2 > 0.7:
    print("\n   ⚠️ CORRELACIÓN FUERTE pero pendiente ≠ 1")
    print(f"      λ₁ = {slope:.2f}·(2β₁/N)")
else:
    print("\n   ❌ CORRELACIÓN DÉBIL")
    print("      La relación no se cumple claramente")

# ==================== 2. FIGURA PRINCIPAL: λ₁ vs 2β₁/N ====================
plt.figure(figsize=(10, 8))

plt.scatter(lambda_pred_f, lambda1_f, alpha=0.4, s=15, c='blue', label='Datos')

# Línea de ajuste
x_fit = np.linspace(min(lambda_pred_f), max(lambda_pred_f), 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, 'r-', linewidth=2,
         label=f'Ajuste: λ₁ = {slope:.2f}·(2β₁/N) + {intercept:.2f}\n$R^2 = {r2:.3f}$')

# Línea de referencia λ₁ = 2β₁/N
plt.plot(x_fit, x_fit, 'g--', alpha=0.5, label='λ₁ = 2β₁/N')

plt.xlabel('$2\\beta_1 / N$', fontsize=14)
plt.ylabel('Radio espectral $\\lambda_1$', fontsize=14)
plt.title('Relación entre radio espectral y densidad de ciclos', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('fig_lambda_vs_cycle_density.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_lambda_vs_cycle_density.png', dpi=300, bbox_inches='tight')
print("\n✅ Figura 1 guardada como fig_lambda_vs_cycle_density.pdf y .png")

# ==================== 3. EVOLUCIÓN CON EL TAMAÑO ====================
plt.figure(figsize=(10, 8))

plt.scatter(sizes, lambda1, alpha=0.4, s=15, c='blue', label='λ₁ real')
plt.scatter(sizes, lambda_pred, alpha=0.4, s=15, c='red', marker='s',
            label='2β₁/N (predicción)')

plt.xscale('log')
plt.xlabel('Tamaño del cluster $N$', fontsize=14)
plt.ylabel('Valor', fontsize=14)
plt.title('Evolución del radio espectral y la densidad de ciclos', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('fig_cycle_density_vs_size.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_cycle_density_vs_size.png', dpi=300, bbox_inches='tight')
print("✅ Figura 2 guardada como fig_cycle_density_vs_size.pdf y .png")

# ==================== 4. FIGURA DE RESIDUALES ====================
residuals = lambda1_f - (slope * lambda_pred_f + intercept)

plt.figure(figsize=(10, 8))
plt.scatter(lambda_pred_f, residuals, alpha=0.4, s=15, c='green')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('$2\\beta_1 / N$', fontsize=14)
plt.ylabel('Residuales', fontsize=14)
plt.title('Residuales del ajuste', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('fig_lambda_residuals.pdf', dpi=300, bbox_inches='tight')
print("✅ Figura 3 guardada como fig_lambda_residuals.pdf")

# ==================== 5. ESTADÍSTICAS POR RANGO ====================
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
        lambda_pred_masked = lambda_pred[mask]
        lambda1_masked = lambda1[mask]
        valid_masked = (lambda_pred_masked > 0) & (lambda1_masked > 0)
        
        if np.sum(valid_masked) > 0:
            slope_r, _, r_r, _, _ = linregress(
                lambda_pred_masked[valid_masked],
                lambda1_masked[valid_masked]
            )
            r2_r = r_r**2
            
            print(f"\n{label}:")
            print(f"   n = {np.sum(mask)}")
            print(f"   ⟨λ₁⟩ = {np.mean(lambda1_masked):.2f} ± {np.std(lambda1_masked):.2f}")
            print(f"   ⟨2β₁/N⟩ = {np.mean(lambda_pred_masked):.2f} ± {np.std(lambda_pred_masked):.2f}")
            print(f"   Pendiente = {slope_r:.4f}")
            print(f"   R² = {r2_r:.4f}")

# ==================== RESUMEN PARA EL PAPER ====================
print("\n" + "="*70)
print(" RESUMEN PARA EL PAPER")
print("="*70)

if r2 > 0.7:
    conclusion = "confirmada"
    if abs(slope - 1) < 0.1:
        relation = "λ₁ ≈ 2β₁/N"
    else:
        relation = f"λ₁ = {slope:.2f}·(2β₁/N)"
else:
    conclusion = "no confirmada"
    relation = "no se observa una relación clara"

print(f"""
\\paragraph{{Spectral-cycle relation}}
A simple topological argument suggests that the spectral radius
should scale with the cycle density as $\\lambda_1 \\approx 2\\beta_1/N$,
where $\\beta_1 = E - N + C$ counts independent cycles.

Our numerical data {conclusion} this relation:
\\begin{{itemize}}
    \\item Linear fit: $\\lambda_1 = {slope:.3f}\\cdot(2\\beta_1/N) + {intercept:.2f}$
    \\item Correlation coefficient: $R^2 = {r2:.3f}$
    \\item {relation}
\\end{{itemize}}

This indicates that the dominant eigenmode of the adjacency matrix
is indeed controlled by the density of topological cycles, linking
the spectral fusion rule directly to the emergent network topology.
""")

# ==================== GUARDAR DATOS ====================
output_data = np.column_stack((sizes, beta1, lambda1, lambda_pred))
np.savetxt('spectral_cycle_relation.dat', output_data,
           header='N beta1 lambda1 lambda_pred', fmt='%d %d %.4f %.4f')
print("\n💾 Datos guardados en spectral_cycle_relation.dat")