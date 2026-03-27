# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:10:23 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radial_curvature_collapse.py

Script 8: Colapso radial de la curvatura de Forman.
- Calcula curvatura media por capa radial.
- Reescala por radio máximo.
- Busca colapso universal y ajusta a familias funcionales.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import glob
import os
import re
import random
import time
import warnings
from collections import defaultdict
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

MIN_NODES = 5000
MAX_GRAPHS = 15
SAMPLE_SIZE = 800

# Archivos de salida
CURVATURE_COLLAPSE_DATA = "radial_curvature_collapse_data.dat"
CURVATURE_FIT_FILE = "radial_curvature_collapse_fit.txt"
SELECTED_GRAPHS_FILE = "selected_graphs_curvature.txt"

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

def approximate_barycentric_center(G, sample_size=SAMPLE_SIZE):
    nodes = list(G.nodes())
    if len(nodes) <= sample_size:
        sample = nodes
    else:
        deg = dict(G.degree())
        high_deg = sorted(deg, key=deg.get, reverse=True)[:sample_size//3]
        rest = [n for n in nodes if n not in high_deg]
        random_rest = random.sample(rest, min(sample_size - len(high_deg), len(rest)))
        sample = high_deg + random_rest
    dist_sums = {}
    for node in sample:
        lengths = nx.single_source_shortest_path_length(G, node)
        dist_sums[node] = sum(lengths.values())
    best = min(dist_sums, key=dist_sums.get)
    return best, dist_sums[best] / len(nodes)

def forman_curvature_edge(G, u, v):
    """Curvatura de Forman para una arista: F = 2 - deg(u) - deg(v) + 2*triangles"""
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    triangles = len(set(G.neighbors(u)) & set(G.neighbors(v)))
    return 2 - deg_u - deg_v + 2 * triangles

def radial_curvature_profile(G, center):
    """Calcula curvatura radial combinada por capa."""
    distances = nx.single_source_shortest_path_length(G, center)
    max_dist = max(distances.values())
    
    # Inicializar acumuladores
    curv_internal_sum = np.zeros(max_dist+1)
    curv_internal_count = np.zeros(max_dist+1)
    curv_radial_sum = np.zeros(max_dist+1)
    curv_radial_count = np.zeros(max_dist+1)
    
    # Para cada arista, clasificar por capas
    for u, v in G.edges():
        du = distances.get(u, -1)
        dv = distances.get(v, -1)
        if du == -1 or dv == -1:
            continue
        
        curv = forman_curvature_edge(G, u, v)
        
        if du == dv:
            curv_internal_sum[du] += curv
            curv_internal_count[du] += 1
        elif abs(du - dv) == 1:
            r_min = min(du, dv)
            curv_radial_sum[r_min] += curv
            curv_radial_count[r_min] += 1
    
    # Curvatura combinada por capa
    curv_combined = np.zeros(max_dist+1)
    for r in range(max_dist+1):
        total_weight = 0
        total_curv = 0
        if curv_internal_count[r] > 0:
            total_curv += curv_internal_sum[r]
            total_weight += curv_internal_count[r]
        if r < max_dist and curv_radial_count[r] > 0:
            total_curv += curv_radial_sum[r]
            total_weight += curv_radial_count[r]
        if total_weight > 0:
            curv_combined[r] = total_curv / total_weight
    
    return curv_combined, max_dist, curv_internal_sum/curv_internal_count, curv_radial_sum/curv_radial_count

# ==================== FUNCIONES DE AJUSTE ====================
def constant_fit(x, a):
    return a * np.ones_like(x)

def linear_fit(x, a, b):
    return a + b * x

def quadratic_fit(x, a, b, c):
    return a + b * x + c * x**2

def cubic_fit(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3

def beta_fit(x, a, b, alpha, beta):
    """Forma beta: a + b * x^alpha * (1-x)^beta"""
    return a + b * (x**alpha) * ((1-x)**beta)

# ==================== PROCESAMIENTO PRINCIPAL ====================
print("="*70)
print(" BLOQUE 8: COLAPSO RADIAL DE CURVATURA")
print("="*70)

# Buscar grafos
graph_files = find_graph_files()
print(f"📁 Total archivos .npz: {len(graph_files)}")

selected = get_large_graphs(graph_files)
print(f"🎯 Seleccionados {len(selected)} grafos con N ≥ {MIN_NODES}")

# Registrar grafos seleccionados
with open(SELECTED_GRAPHS_FILE, 'w') as f:
    f.write("# Selected graphs for curvature collapse analysis\n")
    for fname, N in selected:
        f.write(f"{fname}\t{N}\n")

# Procesar cada grafo
profiles = []  # (N, r, x, curv_combined, max_r)
max_radii = []

print("\n🔬 Calculando perfiles de curvatura radial...")
start_time = time.time()

for idx, (fname, N) in enumerate(selected):
    print(f"   {idx+1}/{len(selected)}: N={N}...", end='', flush=True)
    try:
        A = load_npz(fname)
        G = nx.from_scipy_sparse_array(A)
        G = get_largest_component(G)
        if G.number_of_nodes() < MIN_NODES:
            print("  -> omitido (componente pequeño)")
            continue
        
        # Centro baricéntrico
        center, _ = approximate_barycentric_center(G)
        
        # Perfil de curvatura
        curv_combined, max_r, curv_internal, curv_radial = radial_curvature_profile(G, center)
        
        # Excluir centro (r=0) si produce artefactos
        r = np.arange(max_r+1)
        x = r / max_r
        
        profiles.append({
            'N': N,
            'r': r,
            'x': x,
            'curv': curv_combined,
            'curv_internal': curv_internal,
            'curv_radial': curv_radial,
            'max_r': max_r
        })
        max_radii.append(max_r)
        print(f"  -> max_r={max_r}, curv_centro={curv_combined[1]:.2f}")
        
    except Exception as e:
        print(f"  -> ERROR: {e}")
        continue

elapsed = time.time() - start_time
print(f"\n✅ Procesados {len(profiles)} grafos en {elapsed:.1f} segundos")

if len(profiles) < 3:
    print("❌ Insuficientes grafos para análisis de colapso")
    exit(1)

# ==================== COLAPSO RADIAL ====================
# Grilla común
common_grid = np.linspace(0, 1, 100)

# Interpolar todos los perfiles
all_interp = []
for p in profiles:
    f = interp1d(p['x'], p['curv'], kind='linear', bounds_error=False, fill_value=0)
    interp_vals = f(common_grid)
    all_interp.append(interp_vals)

all_interp = np.array(all_interp)
mean_curv = np.mean(all_interp, axis=0)
std_curv = np.std(all_interp, axis=0)
variance_mean = np.mean(std_curv**2)

print(f"\n📊 Estadísticas del colapso:")
print(f"   Varianza media del colapso: {variance_mean:.6f}")

# ==================== AJUSTE UNIVERSAL ====================
# Probar diferentes familias
fits = {}
r2_scores = {}

# Datos para ajuste (excluir extremos ruidosos)
x_fit = common_grid[5:-5]
y_fit = mean_curv[5:-5]

# Constante
try:
    popt_const, _ = curve_fit(constant_fit, x_fit, y_fit)
    y_pred = constant_fit(x_fit, *popt_const)
    ss_res = np.sum((y_fit - y_pred)**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r2_const = 1 - ss_res/ss_tot
    fits['constant'] = {'popt': popt_const, 'func': constant_fit}
    r2_scores['constant'] = r2_const
except:
    r2_const = -np.inf

# Lineal
try:
    popt_lin, _ = curve_fit(linear_fit, x_fit, y_fit)
    y_pred = linear_fit(x_fit, *popt_lin)
    ss_res = np.sum((y_fit - y_pred)**2)
    r2_lin = 1 - ss_res/ss_tot
    fits['linear'] = {'popt': popt_lin, 'func': linear_fit}
    r2_scores['linear'] = r2_lin
except:
    r2_lin = -np.inf

# Cuadrática
try:
    popt_quad, _ = curve_fit(quadratic_fit, x_fit, y_fit)
    y_pred = quadratic_fit(x_fit, *popt_quad)
    ss_res = np.sum((y_fit - y_pred)**2)
    r2_quad = 1 - ss_res/ss_tot
    fits['quadratic'] = {'popt': popt_quad, 'func': quadratic_fit}
    r2_scores['quadratic'] = r2_quad
except:
    r2_quad = -np.inf

# Cúbica
try:
    popt_cubic, _ = curve_fit(cubic_fit, x_fit, y_fit)
    y_pred = cubic_fit(x_fit, *popt_cubic)
    ss_res = np.sum((y_fit - y_pred)**2)
    r2_cubic = 1 - ss_res/ss_tot
    fits['cubic'] = {'popt': popt_cubic, 'func': cubic_fit}
    r2_scores['cubic'] = r2_cubic
except:
    r2_cubic = -np.inf

# Beta
try:
    popt_beta, _ = curve_fit(beta_fit, x_fit, y_fit, p0=[np.mean(y_fit), np.std(y_fit), 1.0, 1.0], maxfev=5000)
    y_pred = beta_fit(x_fit, *popt_beta)
    ss_res = np.sum((y_fit - y_pred)**2)
    r2_beta = 1 - ss_res/ss_tot
    fits['beta'] = {'popt': popt_beta, 'func': beta_fit}
    r2_scores['beta'] = r2_beta
except:
    r2_beta = -np.inf

# Mejor ajuste
best_fit = max(r2_scores, key=r2_scores.get)
best_popt = fits[best_fit]['popt']
best_func = fits[best_fit]['func']
best_r2 = r2_scores[best_fit]

print(f"\n📈 Mejor ajuste: {best_fit.upper()}")
print(f"   R² = {best_r2:.4f}")
if best_fit == 'constant':
    print(f"   κ(x) = {best_popt[0]:.4f}")
elif best_fit == 'linear':
    print(f"   κ(x) = {best_popt[0]:.4f} + {best_popt[1]:.4f}·x")
elif best_fit == 'quadratic':
    print(f"   κ(x) = {best_popt[0]:.4f} + {best_popt[1]:.4f}·x + {best_popt[2]:.4f}·x²")
elif best_fit == 'cubic':
    print(f"   κ(x) = {best_popt[0]:.4f} + {best_popt[1]:.4f}·x + {best_popt[2]:.4f}·x² + {best_popt[3]:.4f}·x³")
elif best_fit == 'beta':
    print(f"   κ(x) = {best_popt[0]:.4f} + {best_popt[1]:.4f}·x^{best_popt[2]:.4f}·(1-x)^{best_popt[3]:.4f}")

# ==================== GUARDAR DATOS ====================
# Datos colapsados
with open(CURVATURE_COLLAPSE_DATA, 'w') as f:
    f.write("# x mean_curv std_curv\n")
    for i in range(len(common_grid)):
        f.write(f"{common_grid[i]:.6f} {mean_curv[i]:.6f} {std_curv[i]:.6f}\n")

# Parámetros del ajuste
with open(CURVATURE_FIT_FILE, 'w') as f:
    f.write("# Radial curvature collapse analysis\n")
    f.write(f"# Number of graphs: {len(profiles)}\n")
    f.write(f"# Variance mean: {variance_mean:.6f}\n\n")
    f.write(f"Best fit: {best_fit.upper()}\n")
    f.write(f"R² = {best_r2:.4f}\n")
    f.write(f"Parameters: {list(best_popt)}\n")

print(f"💾 Datos guardados en {CURVATURE_COLLAPSE_DATA} y {CURVATURE_FIT_FILE}")

# ==================== FIGURAS ====================
# Figura 12: Perfiles individuales y colapso
fig, ax = plt.subplots(figsize=(10, 8))

# Perfiles individuales en gris
for p in profiles[:10]:  # Mostrar hasta 10 para no saturar
    ax.plot(p['x'], p['curv'], 'gray', alpha=0.3, linewidth=0.8)

# Perfil medio con banda
ax.plot(common_grid, mean_curv, 'b-', linewidth=2, label='Media')
ax.fill_between(common_grid, mean_curv - std_curv, mean_curv + std_curv,
                 color='blue', alpha=0.2, label='±1σ')

ax.set_xlabel('Radio normalizado $r / r_{max}$')
ax.set_ylabel('Curvatura de Forman $\\kappa$')
ax.set_title('Colapso radial de la curvatura')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('fig_radial_curvature_collapse.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_radial_curvature_collapse.png', dpi=300, bbox_inches='tight')
print("✅ Figura 12 guardada: fig_radial_curvature_collapse.pdf/png")

# Figura 13: Ajuste universal
fig, ax = plt.subplots(figsize=(10, 8))

ax.errorbar(common_grid, mean_curv, yerr=std_curv, fmt='o', markersize=3,
            capsize=2, color='blue', alpha=0.6, label='Datos colapsados')

# Curva de ajuste
x_plot = np.linspace(0, 1, 200)
y_plot = best_func(x_plot, *best_popt)
ax.plot(x_plot, y_plot, 'r-', linewidth=2,
        label=f'Ajuste {best_fit.upper()}: $R^2 = {best_r2:.4f}$')

ax.set_xlabel('Radio normalizado $r / r_{max}$')
ax.set_ylabel('Curvatura de Forman $\\kappa$')
ax.set_title('Ajuste universal de la curvatura radial')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('fig_radial_curvature_universal_fit.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_radial_curvature_universal_fit.png', dpi=300, bbox_inches='tight')
print("✅ Figura 13 guardada: fig_radial_curvature_universal_fit.pdf/png")

# ==================== RESUMEN ====================
print("\n" + "="*70)
print(" RESUMEN COLAPSO DE CURVATURA")
print("="*70)
print(f"Grafos analizados: {len(profiles)}")
print(f"Varianza media del colapso: {variance_mean:.6f}")
print(f"Mejor ajuste: {best_fit.upper()} (R² = {best_r2:.4f})")

if variance_mean < 0.1 and best_r2 > 0.7:
    print("\n✅ GOOD CURVATURE COLLAPSE")
    print("   La curvatura radial tiene una forma universal reproducible")
elif variance_mean < 0.3 and best_r2 > 0.5:
    print("\n⚠️ PARTIAL CURVATURE COLLAPSE")
    print("   Hay tendencia a universalidad pero con dispersión")
else:
    print("\n❌ NO CONVINCING CURVATURE UNIVERSALITY")
    print("   La curvatura radial no colapsa de forma universal")

print("\n✅ Script 8 completado.")