# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:49:53 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radial_curvature_and_horizon_test.py

Script 3 del análisis de geometría radial efectiva.
- Calcula curvatura de Forman radial por capas
- Detecta posibles radios críticos (cambios de régimen)
- Simula tiempos de escape desde capas internas
- Identifica posibles "barreras de transporte" o capas tipo horizonte
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.stats import linregress
import glob
import os
import re
import random
import time
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore', category=RuntimeWarning)

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
MAX_GRAPHS = 12  # Reducido para incluir cálculos de escape
SAMPLE_SIZE = 800
N_WALKERS = 200   # Caminantes para tiempo de escape
MAX_STEPS = 500   # Pasos máximos por caminante

# Archivos de salida
CURVATURE_DATA_FILE = "radial_curvature_data.dat"
BARRIER_DATA_FILE = "transport_barrier_data.dat"
HORIZON_FILE = "horizon_candidates.txt"

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

# ==================== CURVATURA DE FORMAN RADIAL ====================
def compute_forman_curvature_edge(G, u, v):
    """Curvatura de Forman para una arista: F = 2 - deg(u) - deg(v) + 2*triangles"""
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    # Número de triángulos = vecinos comunes
    triangles = len(set(G.neighbors(u)) & set(G.neighbors(v)))
    return 2 - deg_u - deg_v + 2 * triangles

def radial_curvature_profile(G, center, max_r=None):
    """Calcula curvatura media por capa radial."""
    distances = nx.single_source_shortest_path_length(G, center)
    max_dist = max(distances.values())
    if max_r is not None:
        max_dist = min(max_dist, max_r)
    
    # Inicializar
    curvature_sum = np.zeros(max_dist+1)
    curvature_count = np.zeros(max_dist+1)
    
    # Para cada arista, determinar sus capas y acumular curvatura
    for u, v in G.edges():
        du = distances.get(u, -1)
        dv = distances.get(v, -1)
        if du == -1 or dv == -1:
            continue
        
        # Solo aristas dentro del rango
        if du <= max_dist and dv <= max_dist:
            # Capa asociada a la arista: la capa más externa de sus extremos
            layer = max(du, dv)
            curv = compute_forman_curvature_edge(G, u, v)
            curvature_sum[layer] += curv
            curvature_count[layer] += 1
    
    # Curvatura media por capa
    curvature_mean = np.zeros(max_dist+1)
    for r in range(max_dist+1):
        if curvature_count[r] > 0:
            curvature_mean[r] = curvature_sum[r] / curvature_count[r]
    
    return curvature_mean, curvature_count

# ==================== TIEMPOS DE ESCAPE ====================
def escape_time_from_layer(G, center, start_layer, target_layer, n_walkers=N_WALKERS, max_steps=MAX_STEPS):
    """
    Simula tiempo medio para alcanzar una capa externa desde una capa interna.
    """
    distances = nx.single_source_shortest_path_length(G, center)
    
    # Nodos en la capa de inicio
    start_nodes = [n for n, d in distances.items() if d == start_layer]
    if not start_nodes:
        return None, 0
    
    # Nodos en la capa objetivo
    target_nodes = set([n for n, d in distances.items() if d >= target_layer])
    
    escape_times = []
    
    for _ in range(n_walkers):
        node = random.choice(start_nodes)
        steps = 0
        while steps < max_steps and node not in target_nodes:
            neighbors = list(G.neighbors(node))
            if neighbors:
                node = random.choice(neighbors)
            steps += 1
        if steps < max_steps:
            escape_times.append(steps)
    
    if escape_times:
        return np.mean(escape_times), len(escape_times)
    else:
        return None, 0

def compute_escape_time_profile(G, center, max_r, n_walkers=N_WALKERS):
    """Calcula tiempo medio de escape desde cada capa interna hasta la capa exterior."""
    escape_times = []
    for r in range(max_r):
        # Tiempo para llegar desde capa r hasta la capa exterior (max_r)
        t, n = escape_time_from_layer(G, center, r, max_r, n_walkers, max_steps=500)
        if t is not None:
            escape_times.append(t)
        else:
            escape_times.append(np.nan)
    return np.array(escape_times)

# ==================== DETECCIÓN DE RADIOS CRÍTICOS ====================
def detect_critical_radii(A_curve, curvature_curve, escape_curve):
    """
    Detecta radios críticos donde ocurren cambios de régimen.
    Criterios:
    - Mínimo local en conectividad (A(r) anómalo)
    - Cambio brusco en curvatura
    - Salto en tiempo de escape
    """
    critical = []
    
    # 1. Cambio en curvatura (gradiente fuerte)
    if len(curvature_curve) > 3:
        grad_curv = np.gradient(curvature_curve)
        for r in range(1, len(grad_curv)-1):
            if abs(grad_curv[r]) > 0.5 * np.std(grad_curv):
                critical.append(('curvature', r))
    
    # 2. Cambio en tiempo de escape
    if len(escape_curve) > 3:
        grad_escape = np.gradient(escape_curve)
        for r in range(1, len(grad_escape)-1):
            if abs(grad_escape[r]) > 0.5 * np.std(grad_escape):
                critical.append(('escape_time', r))
    
    # 3. Cuello en A(r) (mínimo local)
    if len(A_curve) > 3:
        for r in range(1, len(A_curve)-1):
            if A_curve[r] < A_curve[r-1] and A_curve[r] < A_curve[r+1]:
                critical.append(('neck', r))
    
    # Eliminar duplicados y ordenar
    unique = {}
    for t, r in critical:
        if r not in unique:
            unique[r] = t
    return sorted(unique.items())

# ==================== PROCESAMIENTO PRINCIPAL ====================
print("="*70)
print(" BLOQUE 3: CURVATURA RADIAL Y BARRERAS DE TRANSPORTE")
print("="*70)

graph_files = find_graph_files()
print(f"📁 Total archivos .npz: {len(graph_files)}")

selected = get_large_graphs(graph_files)
print(f"🎯 Seleccionados {len(selected)} grafos con N ≥ {MIN_NODES}")
if selected:
    print(f"   Rango: {selected[0][1]} - {selected[-1][1]}")

all_curvature = []
all_escape = []
all_horizons = []

print("\n🔬 Procesando curvatura y tiempos de escape (puede tardar)...")
start_time = time.time()

for idx, (fname, N) in enumerate(selected):
    print(f"   {idx+1}/{len(selected)}: N={N}...", end='', flush=True)
    try:
        A_mat = load_npz(fname)
        G = nx.from_scipy_sparse_array(A_mat)
        G = get_largest_component(G)
        if G.number_of_nodes() < MIN_NODES:
            print("  -> omitido")
            continue
        
        # Centro
        center, _ = approximate_barycentric_center(G)
        
        # Distancias y radio máximo
        distances = nx.single_source_shortest_path_length(G, center)
        max_r = max(distances.values())
        
        # Curvatura radial
        curvature, curv_count = radial_curvature_profile(G, center, max_r)
        
        # Perfiles de capas (necesitamos A(r) para detección de cuellos)
        A_r = np.zeros(max_r+1, dtype=int)
        for d in distances.values():
            if d <= max_r:
                A_r[d] += 1
        
        # Tiempos de escape
        escape_times = compute_escape_time_profile(G, center, max_r, n_walkers=100)
        
        # Detectar radios críticos
        critical = detect_critical_radii(A_r, curvature, escape_times)
        
        # Identificar posible horizonte (primer cambio fuerte)
        horizon_candidate = None
        for t, r in critical:
            if t in ['curvature', 'escape_time'] and r > 0:
                horizon_candidate = r
                break
        
        all_curvature.append({
            'N': N,
            'max_r': max_r,
            'curvature': curvature,
            'curvature_count': curv_count
        })
        
        all_escape.append({
            'N': N,
            'max_r': max_r,
            'escape_times': escape_times,
            'critical': critical,
            'horizon_candidate': horizon_candidate
        })
        
        all_horizons.append({
            'N': N,
            'horizon_candidate': horizon_candidate,
            'n_critical': len(critical),
            'critical_types': [t for t, _ in critical]
        })
        
        print(f"  -> max_r={max_r}, curv_media={np.mean(curvature[1:]):.2f}, horizon_candidate={horizon_candidate}")
        
    except Exception as e:
        print(f"  -> ERROR: {e}")
        continue

elapsed = time.time() - start_time
print(f"\n✅ Procesados {len(all_curvature)} grafos en {elapsed:.1f} segundos")

# ==================== GUARDAR DATOS ====================
# Curvatura radial
with open(CURVATURE_DATA_FILE, 'w') as f:
    f.write("# N max_r mean_curvature std_curvature\n")
    for d in all_curvature:
        curv = d['curvature'][1:]  # excluir centro
        if len(curv) > 0:
            mean_c = np.mean(curv)
            std_c = np.std(curv)
            f.write(f"{d['N']} {d['max_r']} {mean_c:.3f} {std_c:.3f}\n")
        else:
            f.write(f"{d['N']} {d['max_r']} 0.0 0.0\n")

# Guardar perfiles de curvatura individuales
for d in all_curvature:
    out_file = f"curvature_N{d['N']}.dat"
    np.savetxt(out_file, np.column_stack((np.arange(len(d['curvature'])), d['curvature'], d['curvature_count'])),
               header="r curvature count", fmt="%d %.4f %d")
print("💾 Datos de curvatura guardados")

# Barreras de transporte
with open(BARRIER_DATA_FILE, 'w') as f:
    f.write("# N max_r mean_escape_time horizon_candidate n_critical\n")
    for d in all_escape:
        mean_escape = np.nanmean(d['escape_times']) if len(d['escape_times']) > 0 else 0
        f.write(f"{d['N']} {d['max_r']} {mean_escape:.3f} {d['horizon_candidate']} {len(d['critical'])}\n")

# Horizontes candidatos
with open(HORIZON_FILE, 'w') as f:
    f.write("# N horizon_candidate n_critical critical_types\n")
    for d in all_horizons:
        f.write(f"{d['N']} {d['horizon_candidate']} {d['n_critical']} {d['critical_types']}\n")
print(f"💾 Datos guardados en {CURVATURE_DATA_FILE}, {BARRIER_DATA_FILE}, {HORIZON_FILE}")

# ==================== FIGURAS ====================
# Figura 5: Curvatura radial
n_plots = min(4, len(all_curvature))
plot_data = all_curvature[:n_plots]

plt.figure(figsize=(12, 8))
for d in plot_data:
    r = np.arange(len(d['curvature']))
    plt.plot(r, d['curvature'], 'o-', linewidth=1, markersize=3, label=f'N={d["N"]}')
plt.xlabel('Radio r')
plt.ylabel('Curvatura de Forman media')
plt.title('Perfil de curvatura radial')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig('fig_radial_curvature.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_radial_curvature.png', dpi=300, bbox_inches='tight')
print("✅ Figura 5 guardada: fig_radial_curvature.pdf/png")

# Figura 6: Tiempos de escape
n_escape = min(4, len(all_escape))
escape_data = all_escape[:n_escape]

plt.figure(figsize=(12, 8))
for d in escape_data:
    r = np.arange(len(d['escape_times']))
    plt.plot(r, d['escape_times'], 'o-', linewidth=1, markersize=3, label=f'N={d["N"]}')
plt.xlabel('Radio de inicio r')
plt.ylabel('Tiempo medio de escape')
plt.title('Tiempo de escape hasta la capa exterior')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('fig_escape_time_vs_radius.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_escape_time_vs_radius.png', dpi=300, bbox_inches='tight')
print("✅ Figura 6 guardada: fig_escape_time_vs_radius.pdf/png")

# Figura 7: Horizon candidates (solo para grafos con candidatos)
horizon_data = [d for d in all_escape if d['horizon_candidate'] is not None]
if horizon_data:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, d in enumerate(horizon_data[:4]):
        ax = axes[i]
        # Cargar perfiles necesarios (necesitaríamos los datos de geometría)
        # Por simplicidad, mostramos solo curvatura y escape time
        r_curv = np.arange(len(d.get('curvature', [])))
        r_esc = np.arange(len(d.get('escape_times', [])))
        
        ax.plot(r_curv, d.get('curvature', []), 'b-', label='Curvatura')
        ax.plot(r_esc, d.get('escape_times', []) / np.max(d.get('escape_times', [1])), 
                'r-', label='Escape (norm)')
        ax.axvline(x=d['horizon_candidate'], color='k', linestyle='--', 
                   label=f'Horizonte r={d["horizon_candidate"]}')
        ax.set_xlabel('Radio r')
        ax.set_title(f'N={d["N"]}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.suptitle('Candidatos a horizonte: curvatura y tiempo de escape', fontsize=14)
    plt.tight_layout()
    plt.savefig('fig_horizon_candidates.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_horizon_candidates.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 7 guardada: fig_horizon_candidates.pdf/png")
else:
    print("⚠️ No se encontraron candidatos a horizonte para generar Figura 7")

# ==================== RESUMEN ====================
print("\n" + "="*70)
print(" RESUMEN CURVATURA Y BARRERAS")
print("="*70)
print(f"Grafos analizados: {len(all_curvature)}")

mean_curv = np.mean([np.mean(d['curvature'][1:]) for d in all_curvature if len(d['curvature']) > 1])
print(f"Curvatura media (excluyendo centro): {mean_curv:.3f}")

horizon_count = sum(1 for d in all_escape if d['horizon_candidate'] is not None)
print(f"Grafos con candidato a horizonte: {horizon_count}/{len(all_escape)}")

if horizon_count > 0:
    avg_horizon = np.mean([d['horizon_candidate'] for d in all_escape if d['horizon_candidate'] is not None])
    print(f"Radio medio del horizonte candidato: {avg_horizon:.2f}")
    
    if avg_horizon < 5:
        print("   → Horizonte cercano al centro (∼r=3-5)")
    elif avg_horizon < 10:
        print("   → Horizonte en la zona media (∼r=6-9)")
    else:
        print("   → Horizonte en la periferia (∼r>10)")

print("\n✅ Script 3 completado.")