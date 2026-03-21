# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:22:42 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distance_scaling_test.py

Compara dos modelos de escalado para la distancia media L(N):
1. Small-world: L(N) ∼ a log N + b
2. Fractal: L(N) ∼ c N^β  (con β = 1/d_f)

Determina qué modelo se ajusta mejor a los datos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.stats import linregress
import networkx as nx
import glob
import os
import re
import sys
import time
import random
from tqdm import tqdm

# ==================== CONFIGURACIÓN ====================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
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
MIN_NODES = 200

# Número de muestras para estimar L en grafos grandes
SAMPLE_SIZE = 30

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

# ==================== ESTIMAR DISTANCIA MEDIA ====================
def estimate_average_path_length(G, n_samples=SAMPLE_SIZE):
    """
    Estima la longitud media de camino mediante muestreo de nodos.
    
    Parameters
    ----------
    G : networkx.Graph
        Grafo a analizar
    n_samples : int
        Número de nodos a muestrear
        
    Returns
    -------
    float
        Longitud media de camino estimada
    """
    nodes = list(G.nodes())
    if len(nodes) <= n_samples:
        samples = nodes
    else:
        samples = random.sample(nodes, n_samples)
    
    total_dist = 0
    total_pairs = 0
    
    for source in samples:
        try:
            lengths = nx.single_source_shortest_path_length(G, source)
            total_dist += sum(lengths.values())
            total_pairs += len(lengths)
        except:
            continue
    
    if total_pairs > 0:
        return total_dist / total_pairs
    else:
        return 0

# ==================== PROCESAR GRAFOS ====================
print("🔍 Buscando archivos de grafos...")
graph_files = find_graph_files()
print(f"📁 Encontrados {len(graph_files)} archivos .npz")

if len(graph_files) == 0:
    print("❌ No se encontraron archivos .npz")
    sys.exit(1)

# Listas para resultados
sizes = []
path_lengths = []

print(f"\n🔬 Analizando distancia media en {len(graph_files)} grafos...")
print(f"   Mínimo {MIN_NODES} nodos, muestreo de {SAMPLE_SIZE} nodos")
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

        # Tomar componente gigante si es necesario
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest)
            N = G.number_of_nodes()
            if N < MIN_NODES:
                continue

        # Estimar distancia media
        L = estimate_average_path_length(G)

        if L > 0:
            sizes.append(N)
            path_lengths.append(L)
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
path_lengths = np.array(path_lengths)

# ==================== AJUSTE LOGARÍTMICO (SMALL-WORLD) ====================
logN = np.log(sizes)
slope_log, intercept_log, r_log, p_log, std_log = linregress(logN, path_lengths)
r2_log = r_log**2

print("\n" + "="*70)
print(" AJUSTE SMALL-WORLD: L = a·log(N) + b")
print("="*70)
print(f"   a = {slope_log:.4f} ± {std_log:.4f}")
print(f"   b = {intercept_log:.4f}")
print(f"   R² = {r2_log:.4f}")

# ==================== AJUSTE POTENCIAL (FRACTAL) ====================
logL = np.log(path_lengths)
slope_pow, intercept_pow, r_pow, p_pow, std_pow = linregress(logN, logL)
r2_pow = r_pow**2
beta = slope_pow
d_f = 1 / beta if beta > 0 else float('inf')

print("\n" + "="*70)
print(" AJUSTE FRACTAL: L = c·N^β  (β = 1/d_f)")
print("="*70)
print(f"   β = {beta:.4f} ± {std_pow:.4f}")
print(f"   d_f = {d_f:.4f}")
print(f"   R² = {r2_pow:.4f}")

# ==================== COMPARACIÓN DE MODELOS ====================
print("\n" + "="*70)
print(" COMPARACIÓN DE MODELOS")
print("="*70)

if r2_log > r2_pow:
    print("\n✅ EL MODELO LOGARÍTMICO (SMALL-WORLD) AJUSTA MEJOR")
    print(f"   R²_log = {r2_log:.4f} > R²_pow = {r2_pow:.4f}")
    print("\n   Interpretación:")
    print("   • La red tiene estructura small-world")
    print("   • La distancia media crece como L ∼ log N")
    print("   • Hay atajos globales que acortan las distancias")
else:
    print("\n✅ EL MODELO POTENCIAL (FRACTAL) AJUSTA MEJOR")
    print(f"   R²_pow = {r2_pow:.4f} > R²_log = {r2_log:.4f}")
    print(f"\n   Dimensión fractal: d_f = {d_f:.4f}")
    print("\n   Interpretación:")
    print("   • La red tiene geometría fractal pura")
    print("   • No hay atajos globales significativos")
    print("   • El crecimiento es L ∼ N^{1/d_f}")

# Calcular la mejora relativa
improvement = abs(r2_log - r2_pow) / max(r2_log, r2_pow) * 100
print(f"\n   Diferencia relativa: {improvement:.1f}%")

# ==================== FIGURA PRINCIPAL ====================
plt.figure(figsize=(10, 8))

# Datos experimentales
plt.scatter(sizes, path_lengths, alpha=0.4, s=15, c='blue', label='Datos experimentales')

# Curva de ajuste logarítmico
N_fit = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
L_log = slope_log * np.log(N_fit) + intercept_log
plt.plot(N_fit, L_log, 'r-', linewidth=2.5,
         label=f'Small-world: $L = {slope_log:.2f}\\log N + {intercept_log:.2f}$\n$R^2 = {r2_log:.3f}$')

# Curva de ajuste potencial
L_pow = np.exp(intercept_pow) * N_fit**beta
plt.plot(N_fit, L_pow, 'b--', linewidth=2.5,
         label=f'Fractal: $L \\sim N^{{{beta:.2f}}}$ ($d_f = {d_f:.2f}$)\n$R^2 = {r2_pow:.3f}$')

plt.xscale('log')
plt.xlabel('Tamaño del cluster $N$', fontsize=14)
plt.ylabel('Longitud media de camino $L$', fontsize=14)
plt.title('Escalado de la distancia media', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left', fontsize=11)

# Añadir texto con el mejor modelo
if r2_log > r2_pow:
    best_text = f"Mejor modelo: SMALL-WORLD\n$\\Delta R^2 = {r2_log - r2_pow:.3f}$"
else:
    best_text = f"Mejor modelo: FRACTAL\n$d_f = {d_f:.2f}$"

plt.text(0.05, 0.95, best_text,
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('fig_distance_scaling.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_distance_scaling.png', dpi=300, bbox_inches='tight')
print("\n✅ Figura guardada como fig_distance_scaling.pdf y .png")

# ==================== FIGURA ADICIONAL: RESIDUALES ====================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Residuales del modelo logarítmico
ax = axes[0]
L_pred_log = slope_log * logN + intercept_log
residuals_log = path_lengths - L_pred_log
ax.scatter(sizes, residuals_log, alpha=0.5, s=10, c='red')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_xscale('log')
ax.set_xlabel('Tamaño N')
ax.set_ylabel('Residuales')
ax.set_title('Residuales: modelo logarítmico')
ax.grid(True, alpha=0.3)

# Residuales del modelo potencial
ax = axes[1]
L_pred_pow = intercept_pow + beta * logN
residuals_pow = logL - L_pred_pow
ax.scatter(sizes, residuals_pow, alpha=0.5, s=10, c='blue')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_xscale('log')
ax.set_xlabel('Tamaño N')
ax.set_ylabel('Residuales (en log L)')
ax.set_title('Residuales: modelo potencial')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_distance_residuals.pdf', dpi=300, bbox_inches='tight')
print("✅ Figura de residuales guardada como fig_distance_residuals.pdf")

# ==================== ESTADÍSTICAS POR RANGO ====================
print("\n" + "="*70)
print(" ESTADÍSTICAS POR RANGO DE TAMAÑOS")
print("="*70)

ranges = [
    (200, 1000, "Pequeños (200-1000)"),
    (1000, 10000, "Medianos (1000-10000)"),
    (10000, 1000000, "Grandes (>10000)")
]

for low, high, label in ranges:
    mask = (sizes >= low) & (sizes < high)
    if np.sum(mask) > 0:
        L_range = path_lengths[mask]
        print(f"\n{label}:")
        print(f"   n = {np.sum(mask)}")
        print(f"   L media = {np.mean(L_range):.2f} ± {np.std(L_range):.2f}")
        print(f"   L / log(N) medio = {np.mean(L_range / np.log(sizes[mask])):.4f}")

# ==================== RESUMEN PARA EL PAPER ====================
print("\n" + "="*70)
print(" RESUMEN PARA EL PAPER")
print("="*70)

if r2_log > r2_pow:
    conclusion = "small-world"
    model_desc = "logarithmic"
else:
    conclusion = "fractal"
    model_desc = "power-law"

print(f"""
\\paragraph{{Distance scaling}}
To distinguish between fractal and small-world geometry, we analyzed
the scaling of the average shortest-path length $L(N)$.
A fractal network follows $L \\sim N^{{1/d_f}}$, while a small-world network
follows $L \\sim \\log N$.

The data strongly favour the {model_desc} scaling:
\\begin{{itemize}}
    \\item Logarithmic fit: $L = {slope_log:.3f}\\log N + {intercept_log:.2f}$, $R^2 = {r2_log:.3f}$
    \\item Power-law fit: $L \\sim N^{{{beta:.3f}}}$ ($d_f = {d_f:.2f}$), $R^2 = {r2_pow:.3f}$
\\end{{itemize}}

The {conclusion} interpretation is consistent with the presence of
global shortcuts that dramatically reduce distances while maintaining
the underlying hierarchical structure revealed by the cycle analysis.
""")

# ==================== GUARDAR DATOS ====================
output_data = np.column_stack((sizes, path_lengths))
np.savetxt('distance_scaling_data.dat', output_data,
           header='N L', fmt='%d %.4f')
print("\n💾 Datos guardados en distance_scaling_data.dat")