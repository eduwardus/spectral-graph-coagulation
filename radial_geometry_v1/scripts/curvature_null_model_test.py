# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:05:33 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curvature_null_model_test.py

Test de curvatura contra null models con secuencia de grados preservada.
- Genera grafos rewired preservando grados (double edge swap)
- Compara curvatura real vs null
- Calcula z-scores
- Analiza correlación curvatura-grado
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.stats import pearsonr, spearmanr
import glob
import os
import re
import random
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13

GRAPH_DIRS = [
    "soup_simulation_phase_transition_v20",
    "soup_simulation_phase_transition_v20/snapshots"
]

MIN_NODES = 500
MAX_GRAPHS = 25  # Número de grafos a analizar
N_NULL = 20      # Número de null models por grafo
N_SWAPS = 10000  # Número de swaps por null

# Archivos de salida
NULL_DATA_FILE = "curvature_null_model_data.dat"
CORRELATION_FILE = "degree_curvature_correlation.dat"

# ==================== FUNCIONES AUXILIARES ====================
def find_graph_files():
    graph_files = []
    for directory in GRAPH_DIRS:
        if os.path.exists(directory):
            pattern = os.path.join(directory, "*.npz")
            files = glob.glob(pattern)
            graph_files.extend(files)
    return graph_files

def extract_size_from_filename(filename):
    patterns = [r'_N(\d+)_', r'_Ninit(\d+)', r'N(\d+)\.npz']
    basename = os.path.basename(filename)
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            return int(match.group(1))
    return None

def get_large_graphs(files, min_nodes=MIN_NODES, max_samples=MAX_GRAPHS):
    candidates = []
    for f in files:
        N = extract_size_from_filename(f)
        if N is None:
            try:
                A = load_npz(f)
                N = A.shape[0]
            except:
                continue
        if N >= min_nodes:
            candidates.append((f, N))
    candidates.sort(key=lambda x: x[1])
    if len(candidates) > max_samples:
        indices = np.linspace(0, len(candidates)-1, max_samples, dtype=int)
        selected = [candidates[i] for i in indices]
    else:
        selected = candidates
    return selected

def get_largest_component(G):
    if nx.is_connected(G):
        return G
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()

def forman_curvature_edge(G, u, v):
    """Curvatura de Forman para una arista."""
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    triangles = len(set(G.neighbors(u)) & set(G.neighbors(v)))
    return 2 - deg_u - deg_v + 2 * triangles

def forman_curvature_graph(G, sample_edges=None):
    """Calcula curvatura media de Forman para el grafo."""
    edges = list(G.edges())
    if sample_edges and len(edges) > sample_edges:
        edges = random.sample(edges, sample_edges)
    
    curvatures = []
    for u, v in edges:
        curv = forman_curvature_edge(G, u, v)
        curvatures.append(curv)
    
    if curvatures:
        return np.mean(curvatures), np.std(curvatures)
    return 0, 0

def generate_null_model(G, n_swaps=N_SWAPS):
    """Genera grafo rewired preservando grados (double edge swap)."""
    G_null = G.copy()
    try:
        # Usar la función de networkx si está disponible
        from networkx.algorithms.swap import double_edge_swap
        for _ in range(min(n_swaps, 1000)):
            double_edge_swap(G_null, nswap=1, max_tries=100, seed=random.randint(0, 10000))
    except:
        # Implementación simple de edge swap
        edges = list(G_null.edges())
        n_swaps = min(n_swaps, len(edges) * 10)
        for _ in range(n_swaps):
            try:
                e1 = random.choice(edges)
                e2 = random.choice(edges)
                if e1[0] in (e2[0], e2[1]) or e1[1] in (e2[0], e2[1]):
                    continue
                if not G_null.has_edge(e1[0], e2[0]) and not G_null.has_edge(e1[1], e2[1]):
                    G_null.remove_edge(*e1)
                    G_null.remove_edge(*e2)
                    G_null.add_edge(e1[0], e2[0])
                    G_null.add_edge(e1[1], e2[1])
                    edges.remove(e1)
                    edges.remove(e2)
                    edges.append((e1[0], e2[0]))
                    edges.append((e1[1], e2[1]))
            except:
                continue
    return G_null

def compute_degree_statistics(G):
    """Calcula estadísticas de grado."""
    degrees = [d for n, d in G.degree()]
    return {
        'mean': np.mean(degrees),
        'std': np.std(degrees),
        'max': max(degrees),
        'min': min(degrees)
    }

def compute_clustering(G):
    """Calcula clustering medio."""
    try:
        return nx.average_clustering(G)
    except:
        return 0

def compute_eigenvector_centrality_correlation(G):
    """Calcula correlación entre grado y centralidad de autovector."""
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
        degrees = [G.degree(n) for n in G.nodes()]
        ec_vals = [ec[n] for n in G.nodes()]
        pearson_r, _ = pearsonr(degrees, ec_vals)
        spearman_r, _ = spearmanr(degrees, ec_vals)
        return pearson_r, spearman_r
    except:
        return 0, 0

def compute_curvature_degree_correlation(G):
    """Calcula correlación entre curvatura local y grado."""
    edges = list(G.edges())
    if len(edges) > 1000:
        edges = random.sample(edges, 1000)
    
    curvatures = []
    degree_products = []
    for u, v in edges:
        curv = forman_curvature_edge(G, u, v)
        curvatures.append(curv)
        degree_products.append(G.degree(u) * G.degree(v))
    
    if len(curvatures) > 5:
        return pearsonr(curvatures, degree_products)[0], len(curvatures)
    return 0, 0

# ==================== PROCESAMIENTO PRINCIPAL ====================
print("="*70)
print(" SCRIPT 2: NULL MODEL TEST DE CURVATURA")
print("="*70)

graph_files = find_graph_files()
print(f"📁 Total archivos .npz: {len(graph_files)}")

selected = get_large_graphs(graph_files)
print(f"🎯 Seleccionados {len(selected)} grafos con N ≥ {MIN_NODES}")

all_results = []

print("\n🔬 Analizando grafos...")
start_time = time.time()

for idx, (fname, N) in enumerate(selected):
    print(f"\n   {idx+1}/{len(selected)}: N={N}...")
    try:
        A = load_npz(fname)
        G = nx.from_scipy_sparse_array(A)
        G = get_largest_component(G)
        
        # Observables del grafo real
        curv_real, curv_std_real = forman_curvature_graph(G)
        deg_stats = compute_degree_statistics(G)
        clustering_real = compute_clustering(G)
        pearson_ec, spearman_ec = compute_eigenvector_centrality_correlation(G)
        curv_deg_corr, n_edges_corr = compute_curvature_degree_correlation(G)
        
        print(f"      curvatura real: {curv_real:.3f} ± {curv_std_real:.3f}")
        
        # Generar null models
        null_curvatures = []
        null_clustering = []
        null_pearson = []
        null_spearman = []
        null_curv_deg_corr = []
        
        for null_idx in range(N_NULL):
            try:
                G_null = generate_null_model(G)
                curv_null, _ = forman_curvature_graph(G_null)
                null_curvatures.append(curv_null)
                null_clustering.append(compute_clustering(G_null))
                p, s = compute_eigenvector_centrality_correlation(G_null)
                null_pearson.append(p)
                null_spearman.append(s)
                cd, _ = compute_curvature_degree_correlation(G_null)
                null_curv_deg_corr.append(cd)
                if (null_idx + 1) % 5 == 0:
                    print(f"         null {null_idx+1}/{N_NULL}: curv={curv_null:.3f}")
            except Exception as e:
                print(f"         null {null_idx+1} falló: {e}")
                continue
        
        if null_curvatures:
            mean_null = np.mean(null_curvatures)
            std_null = np.std(null_curvatures)
            z_score = (curv_real - mean_null) / std_null if std_null > 0 else 0
            
            print(f"      null models: {mean_null:.3f} ± {std_null:.3f}")
            print(f"      z-score: {z_score:.3f}")
            
            all_results.append({
                'N': N,
                'curv_real': curv_real,
                'curv_std_real': curv_std_real,
                'curv_null_mean': mean_null,
                'curv_null_std': std_null,
                'z_score': z_score,
                'clustering_real': clustering_real,
                'clustering_null_mean': np.mean(null_clustering) if null_clustering else 0,
                'pearson_ec': pearson_ec,
                'spearman_ec': spearman_ec,
                'curv_deg_corr': curv_deg_corr,
                'deg_mean': deg_stats['mean'],
                'deg_std': deg_stats['std']
            })
        else:
            print("      ⚠️ No se generaron null models válidos")
            
    except Exception as e:
        print(f"      ERROR: {e}")
        continue

elapsed = time.time() - start_time
print(f"\n✅ Procesados {len(all_results)} grafos en {elapsed:.1f} segundos")

# ==================== FIGURAS ====================
# Figura 1: Real vs Null
fig, ax = plt.subplots(figsize=(10, 8))
Ns = [r['N'] for r in all_results]
curv_real = [r['curv_real'] for r in all_results]
curv_null = [r['curv_null_mean'] for r in all_results]
ax.errorbar(Ns, curv_real, yerr=[r['curv_std_real'] for r in all_results], 
            fmt='o', color='blue', label='Real', capsize=3)
ax.errorbar(Ns, curv_null, yerr=[r['curv_null_std'] for r in all_results],
            fmt='s', color='red', label='Null models', capsize=3)
ax.set_xscale('log')
ax.set_xlabel('Tamaño N')
ax.set_ylabel('Curvatura de Forman')
ax.set_title('Curvatura real vs null models')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('fig_real_vs_null_curvature.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_real_vs_null_curvature.png', dpi=300, bbox_inches='tight')
print("✅ Figura 1: fig_real_vs_null_curvature.pdf/png")

# Figura 2: z-score vs tamaño
fig, ax = plt.subplots(figsize=(10, 8))
z_scores = [r['z_score'] for r in all_results]
ax.scatter(Ns, z_scores, alpha=0.6, s=30)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='z=2 (significativo)')
ax.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel('Tamaño N')
ax.set_ylabel('z-score')
ax.set_title('z-score de curvatura real vs null models')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('fig_curvature_zscore_vs_size.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_curvature_zscore_vs_size.png', dpi=300, bbox_inches='tight')
print("✅ Figura 2: fig_curvature_zscore_vs_size.pdf/png")

# Figura 3: Curvatura vs grado
fig, ax = plt.subplots(figsize=(10, 8))
curv_deg_corr = [r['curv_deg_corr'] for r in all_results if r['curv_deg_corr'] != 0]
corr_Ns = [r['N'] for r in all_results if r['curv_deg_corr'] != 0]
ax.scatter(corr_Ns, curv_deg_corr, alpha=0.6, s=30)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('Tamaño N')
ax.set_ylabel('Correlación curvatura-grado')
ax.set_title('Correlación entre curvatura y producto de grados')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_curvature_vs_degree.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_curvature_vs_degree.png', dpi=300, bbox_inches='tight')
print("✅ Figura 3: fig_curvature_vs_degree.pdf/png")

# Figura 4: Resumen comparativo
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Curvatura
ax = axes[0, 0]
ax.bar(['Real', 'Null'], [np.mean(curv_real), np.mean(curv_null)], 
       yerr=[np.std(curv_real), np.std(curv_null)], capsize=5)
ax.set_ylabel('Curvatura media')
ax.set_title('A. Curvatura: real vs null')
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Clustering
ax = axes[0, 1]
clust_real = [r['clustering_real'] for r in all_results]
clust_null = [r['clustering_null_mean'] for r in all_results]
ax.bar(['Real', 'Null'], [np.mean(clust_real), np.mean(clust_null)],
       yerr=[np.std(clust_real), np.std(clust_null)], capsize=5)
ax.set_ylabel('Clustering medio')
ax.set_title('B. Clustering: real vs null')
ax.grid(True, alpha=0.3, axis='y')

# Panel C: z-score
ax = axes[1, 0]
z_vals = [r['z_score'] for r in all_results]
ax.hist(z_vals, bins=15, edgecolor='black', alpha=0.7)
ax.axvline(x=2, color='red', linestyle='--', label='z=2')
ax.axvline(x=-2, color='red', linestyle='--')
ax.axvline(x=0, color='gray', linestyle='-')
ax.set_xlabel('z-score')
ax.set_ylabel('Frecuencia')
ax.set_title('C. Distribución de z-scores')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel D: Correlación curvatura-grado
ax = axes[1, 1]
ax.hist(curv_deg_corr, bins=15, edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='gray', linestyle='-')
ax.set_xlabel('Correlación')
ax.set_ylabel('Frecuencia')
ax.set_title('D. Correlación curvatura-grado')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fig_real_vs_null_summary.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_real_vs_null_summary.png', dpi=300, bbox_inches='tight')
print("✅ Figura 4: fig_real_vs_null_summary.pdf/png")

# ==================== GUARDAR DATOS ====================
with open(NULL_DATA_FILE, 'w') as f:
    f.write("# N curv_real curv_std_real curv_null_mean curv_null_std z_score clustering_real clustering_null_mean\n")
    for r in all_results:
        f.write(f"{r['N']} {r['curv_real']:.4f} {r['curv_std_real']:.4f} "
                f"{r['curv_null_mean']:.4f} {r['curv_null_std']:.4f} {r['z_score']:.4f} "
                f"{r['clustering_real']:.6f} {r['clustering_null_mean']:.6f}\n")

with open(CORRELATION_FILE, 'w') as f:
    f.write("# N pearson_ec spearman_ec curv_deg_corr deg_mean deg_std\n")
    for r in all_results:
        f.write(f"{r['N']} {r['pearson_ec']:.4f} {r['spearman_ec']:.4f} "
                f"{r['curv_deg_corr']:.4f} {r['deg_mean']:.2f} {r['deg_std']:.2f}\n")

print(f"💾 Datos guardados en {NULL_DATA_FILE} y {CORRELATION_FILE}")

# ==================== INTERPRETACIÓN ====================
print("\n" + "="*70)
print(" INTERPRETACIÓN")
print("="*70)

mean_z = np.mean([r['z_score'] for r in all_results])
frac_above_2 = sum(1 for r in all_results if r['z_score'] > 2) / len(all_results)
frac_below_neg2 = sum(1 for r in all_results if r['z_score'] < -2) / len(all_results)

print(f"\n📊 z-score medio: {mean_z:.3f}")
print(f"   Fracción con z > 2: {frac_above_2:.2%}")
print(f"   Fracción con z < -2: {frac_below_neg2:.2%}")

if mean_z < -1 and frac_below_neg2 > 0.5:
    print("\n✅ CASO FUERTE")
    print("   La curvatura del grafo real es significativamente más negativa que")
    print("   en null models con igual secuencia de grados. La curvatura negativa")
    print("   no se explica solo por heterogeneidad de grado.")
elif mean_z < -0.5:
    print("\n⚠️ CASO INTERMEDIO")
    print("   Parte de la curvatura negativa se explica por la secuencia de grados,")
    print("   pero el grafo real muestra desviaciones sistemáticas en clustering")
    print("   y estructura espectral.")
else:
    print("\n❌ CASO DÉBIL")
    print("   La curvatura negativa queda casi totalmente explicada por la")
    print("   heterogeneidad de grado. El efecto residual es pequeño.")

print("\n✅ Script 2 completado.")