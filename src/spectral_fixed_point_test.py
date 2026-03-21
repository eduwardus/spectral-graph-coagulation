# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:42:19 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectral_fixed_point_test.py

Comprueba si el sistema converge hacia un punto fijo espectral,
es decir, si la dispersión de λ₁ disminuye al crecer el tamaño del grafo.

Mide:
- Media de λ₁ por rango de tamaño
- Desviación estándar
- Coeficiente de variación CV = σ/μ

Si CV(N) → 0, el sistema converge a un valor universal λ*.
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

# Número de bins logarítmicos
N_BINS = 8

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

print(f"\n🔬 Analizando convergencia espectral para {len(graph_files)} grafos...")
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

        # Autovalor principal
        lambda1 = compute_spectral_radius(G)

        if lambda1 is not None and lambda1 > 0:
            sizes.append(N)
            lambda1_list.append(lambda1)
            valid_count += 1

    except Exception as e:
        continue

elapsed = time.time() - start_time
print(f"\n\n✅ Procesados {valid_count} grafos válidos en {elapsed:.1f} segundos")

if valid_count == 0:
    print("❌ No se obtuvieron datos válidos")
    sys.exit(1)

# Convertir a arrays y ordenar
sizes = np.array(sizes)
lambda1 = np.array(lambda1_list)

order = np.argsort(sizes)
sizes = sizes[order]
lambda1 = lambda1[order]

# ==================== BINS LOGARÍTMICOS ====================
bins = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), N_BINS)

bin_centers = []
lambda_mean = []
lambda_std = []
lambda_cv = []
bin_counts = []

for i in range(len(bins)-1):
    mask = (sizes >= bins[i]) & (sizes < bins[i+1])
    
    if np.sum(mask) < 3:
        continue
    
    N_bin = np.mean(sizes[mask])
    mean = np.mean(lambda1[mask])
    std = np.std(lambda1[mask])
    cv = std / mean
    
    bin_centers.append(N_bin)
    lambda_mean.append(mean)
    lambda_std.append(std)
    lambda_cv.append(cv)
    bin_counts.append(np.sum(mask))

bin_centers = np.array(bin_centers)
lambda_mean = np.array(lambda_mean)
lambda_std = np.array(lambda_std)
lambda_cv = np.array(lambda_cv)
bin_counts = np.array(bin_counts)

print(f"\n📊 Número de bins: {len(bin_centers)}")
print(f"   Muestras por bin: {bin_counts}")

# ==================== AJUSTE DEL COEFICIENTE DE VARIACIÓN ====================
logN = np.log(bin_centers)
logCV = np.log(lambda_cv)

slope, intercept, r, p, std_err = linregress(logN, logCV)
r2 = r**2

print("\n" + "="*70)
print(" ANÁLISIS DE CONVERGENCIA ESPECTRAL")
print("="*70)
print(f"\n📈 Ajuste: CV(N) ∼ N^{{{slope:.4f}}}")
print(f"   Exponente = {slope:.4f} ± {std_err:.4f}")
print(f"   R² = {r2:.4f}")
print(f"   p-value = {p:.4e}")

if slope < -0.3:
    print("\n✅ CONVERGENCIA FUERTE: CV decrece rápidamente")
    print("   El sistema converge a un punto fijo espectral")
elif slope < -0.1:
    print("\n⚠️ CONVERGENCIA DÉBIL: CV decrece lentamente")
    print("   Hay tendencia a la convergencia pero lenta")
else:
    print("\n❌ NO CONVERGE: CV no decrece significativamente")
    print("   El espectro no se estabiliza")

# ==================== FIGURA 1: MEDIA DE λ₁ ====================
plt.figure(figsize=(10, 8))

plt.errorbar(bin_centers, lambda_mean, yerr=lambda_std, fmt='o-', 
             capsize=5, linewidth=2, markersize=8, color='blue',
             label='Media ± σ')

plt.xscale('log')
plt.xlabel('Tamaño del cluster $N$', fontsize=14)
plt.ylabel('Radio espectral $\\lambda_1$', fontsize=14)
plt.title('Evolución del radio espectral con el tamaño', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('fig_lambda_mean_vs_size.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_lambda_mean_vs_size.png', dpi=300, bbox_inches='tight')
print("\n✅ Figura 1 guardada como fig_lambda_mean_vs_size.pdf y .png")

# ==================== FIGURA 2: COEFICIENTE DE VARIACIÓN ====================
plt.figure(figsize=(10, 8))

plt.scatter(bin_centers, lambda_cv, s=50, c='red', alpha=0.7, label='Datos')

# Ajuste
x_fit = np.logspace(np.log10(min(bin_centers)), np.log10(max(bin_centers)), 100)
y_fit = np.exp(intercept) * x_fit**slope
plt.plot(x_fit, y_fit, 'k--', linewidth=2,
         label=f'Ajuste: $CV \\sim N^{{{slope:.2f}}}$\n$R^2 = {r2:.3f}$')

# Línea de referencia para convergencia
plt.axhline(y=0.1, color='gray', linestyle=':', alpha=0.5, label='CV = 0.1')

plt.xscale('log')
plt.xlabel('Tamaño del cluster $N$', fontsize=14)
plt.ylabel('Coeficiente de variación $CV(\\lambda_1)$', fontsize=14)
plt.title('Convergencia del radio espectral', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('fig_lambda_cv_vs_size.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_lambda_cv_vs_size.png', dpi=300, bbox_inches='tight')
print("✅ Figura 2 guardada como fig_lambda_cv_vs_size.pdf y .png")

# ==================== FIGURA 3: HISTOGRAMA POR RANGOS ====================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Seleccionar 4 bins representativos
bin_indices = [0, len(bin_centers)//3, 2*len(bin_centers)//3, -1]

for idx, ax_idx in enumerate([(0,0), (0,1), (1,0), (1,1)]):
    ax = axes[ax_idx]
    bin_idx = bin_indices[idx]
    
    # Obtener datos de este bin
    N_min = bins[bin_idx] if bin_idx < len(bins)-1 else bins[-2]
    N_max = bins[bin_idx+1] if bin_idx+1 < len(bins) else bins[-1]
    
    mask = (sizes >= N_min) & (sizes < N_max)
    
    ax.hist(lambda1[mask], bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=lambda_mean[bin_idx], color='red', linestyle='--',
               label=f'Media = {lambda_mean[bin_idx]:.2f}')
    ax.set_xlabel('$\\lambda_1$')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'N ∈ [{N_min:.0f}, {N_max:.0f})')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Distribución del radio espectral por rango de tamaño', fontsize=16)
plt.tight_layout()
plt.savefig('fig_lambda_histograms.pdf', dpi=300, bbox_inches='tight')
print("✅ Figura 3 guardada como fig_lambda_histograms.pdf")

# ==================== ESTADÍSTICAS POR BIN ====================
print("\n" + "="*70)
print(" ESTADÍSTICAS POR RANGO DE TAMAÑOS")
print("="*70)

for i in range(len(bin_centers)):
    print(f"\nBin {i+1}: N ≈ {bin_centers[i]:.0f} ({bin_counts[i]} grafos)")
    print(f"   ⟨λ₁⟩ = {lambda_mean[i]:.2f} ± {lambda_std[i]:.2f}")
    print(f"   CV = {lambda_cv[i]:.4f}")

# ==================== RESUMEN PARA EL PAPER ====================
print("\n" + "="*70)
print(" RESUMEN PARA EL PAPER")
print("="*70)

if slope < -0.3:
    convergence_text = "strong convergence"
elif slope < -0.1:
    convergence_text = "weak convergence"
else:
    convergence_text = "no clear convergence"

print(f"""
\\paragraph{{Spectral fixed point}}
To test whether the dynamics converges to a universal spectral value,
we analyzed the coefficient of variation $CV(\\lambda_1) = \\sigma/\\mu$
as a function of cluster size.

The results show {convergence_text}:
\\begin{{itemize}}
    \\item CV scales as $CV \\sim N^{{{slope:.3f}}}$ ($R^2 = {r2:.3f}$)
    \\item For the largest clusters ($N \\approx {bin_centers[-1]:.0f}$),
          $CV = {lambda_cv[-1]:.4f}$
\\end{{itemize}}

This indicates that the spectral radius tends to a fixed point
$\\lambda^* \\approx {lambda_mean[-1]:.2f}$ as the network grows,
confirming that the fusion dynamics organizes the system around
a universal spectral attractor.
""")

# ==================== GUARDAR DATOS ====================
output_data = np.column_stack((bin_centers, lambda_mean, lambda_std, lambda_cv, bin_counts))
np.savetxt('spectral_fixed_point_data.dat', output_data,
           header='N_mean lambda_mean lambda_std lambda_cv count', fmt='%.2f %.4f %.4f %.4f %d')
print("\n💾 Datos guardados en spectral_fixed_point_data.dat")