# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:10:56 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transport_potential_collapse.py

Script 9: Colapso radial del potencial efectivo de transporte.
- Calcula U_eff(r) = log(T_escape(r)/T_escape(R_ref)).
- Reescala por radio máximo.
- Prueba dos normalizaciones de amplitud.
- Busca colapso universal y ajusta a familias funcionales.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import networkx as nx
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

MIN_NODES = 5000
MAX_GRAPHS = 12
SAMPLE_SIZE = 800
N_WALKERS = 200
MAX_STEPS = 500

# Archivos de salida
POTENTIAL_DATA_FILE = "transport_potential_collapse_data.dat"
POTENTIAL_FIT_FILE = "transport_potential_collapse_fit.txt"
SELECTED_GRAPHS_FILE = "selected_graphs_potential.txt"

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

def escape_time_from_layer(G, center, start_layer, target_layer, n_walkers=100, max_steps=500):
    """Simula tiempo medio de escape desde una capa interna hasta capa externa."""
    distances = nx.single_source_shortest_path_length(G, center)
    start_nodes = [n for n, d in distances.items() if d == start_layer]
    target_nodes = set([n for n, d in distances.items() if d >= target_layer])
    
    if not start_nodes or not target_nodes:
        return None, 0
    
    escape_times = []
    for _ in range(min(n_walkers, len(start_nodes))):
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
    return None, 0

def compute_escape_profile(G, center, max_r):
    """Calcula perfil completo de tiempos de escape."""
    escape_times = []
    for r in range(max_r):
        t, n = escape_time_from_layer(G, center, r, max_r, N_WALKERS, MAX_STEPS)
        if t is not None:
            escape_times.append(t)
        else:
            escape_times.append(np.nan)
    return np.array(escape_times)

# ==================== FUNCIONES DE AJUSTE ====================
def linear_fit(x, a, b):
    return a + b * x

def quadratic_fit(x, a, b, c):
    return a + b * x + c * x**2

def cubic_fit(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3

def beta_fit(x, a, b, alpha, beta):
    return a + b * (x**alpha) * ((1-x)**beta)

def combined_fit(x, a, b, c):
    """Cuadrática con offset: a + b*x + c*x²"""
    return a + b * x + c * x**2

# ==================== PROCESAMIENTO PRINCIPAL ====================
print("="*70)
print(" BLOQUE 9: COLAPSO RADIAL DEL POTENCIAL DE TRANSPORTE")
print("="*70)

graph_files = find_graph_files()
print(f"📁 Total archivos .npz: {len(graph_files)}")

selected = get_large_graphs(graph_files)
print(f"🎯 Seleccionados {len(selected)} grafos con N ≥ {MIN_NODES}")

# Registrar grafos seleccionados
with open(SELECTED_GRAPHS_FILE, 'w') as f:
    f.write("# Selected graphs for transport potential analysis\n")
    for fname, N in selected:
        f.write(f"{fname}\t{N}\n")

# Procesar cada grafo
profiles_raw = []   # (N, r, x, U_eff, max_r)
profiles_norm = []  # (N, r, x, U_norm, max_r)

print("\n🔬 Calculando perfiles de transporte (puede tardar)...")
start_time = time.time()

for idx, (fname, N) in enumerate(selected):
    print(f"   {idx+1}/{len(selected)}: N={N}...", end='', flush=True)
    try:
        A = load_npz(fname)
        G = nx.from_scipy_sparse_array(A)
        G = get_largest_component(G)
        if G.number_of_nodes() < MIN_NODES:
            print("  -> omitido")
            continue
        
        center, _ = approximate_barycentric_center(G)
        distances = nx.single_source_shortest_path_length(G, center)
        max_r = max(distances.values())
        
        # Calcular tiempos de escape
        escape_times = compute_escape_profile(G, center, max_r)
        
        # Calcular potencial efectivo
        valid = ~np.isnan(escape_times)
        if np.sum(valid) < 3:
            print("  -> pocos tiempos válidos")
            continue
        
        # Referencia: última capa con escape válido
        ref_idx = max([i for i, v in enumerate(escape_times) if not np.isnan(v)])
        ref_time = escape_times[ref_idx]
        
        U_eff = np.log(escape_times / ref_time)
        
        r = np.arange(max_r)
        x = r / max_r
        
        # Normalización de amplitud
        U_max = np.max(np.abs(U_eff[valid]))
        U_norm = U_eff / U_max
        
        profiles_raw.append({
            'N': N,
            'r': r,
            'x': x,
            'U': U_eff,
            'max_r': max_r
        })
        
        profiles_norm.append({
            'N': N,
            'r': r,
            'x': x,
            'U': U_norm,
            'max_r': max_r
        })
        
        print(f"  -> max_r={max_r}, U_eff_range=[{np.nanmin(U_eff):.2f}, {np.nanmax(U_eff):.2f}]")
        
    except Exception as e:
        print(f"  -> ERROR: {e}")
        continue

elapsed = time.time() - start_time
print(f"\n✅ Procesados {len(profiles_raw)} grafos en {elapsed:.1f} segundos")

if len(profiles_raw) < 3:
    print("❌ Insuficientes grafos para análisis de colapso")
    exit(1)

# ==================== COLAPSO RADIAL ====================
common_grid = np.linspace(0, 1, 100)

def collapse_profiles(profiles):
    all_interp = []
    for p in profiles:
        valid = ~np.isnan(p['U'])
        if np.sum(valid) < 3:
            continue
        f = interp1d(p['x'][valid], p['U'][valid], kind='linear', bounds_error=False, fill_value=0)
        interp_vals = f(common_grid)
        all_interp.append(interp_vals)
    if not all_interp:
        return None, None, None
    all_interp = np.array(all_interp)
    return np.mean(all_interp, axis=0), np.std(all_interp, axis=0), all_interp

# Colapso sin normalización
mean_raw, std_raw, _ = collapse_profiles(profiles_raw)
variance_raw = np.mean(std_raw**2) if std_raw is not None else np.inf

# Colapso con normalización
mean_norm, std_norm, _ = collapse_profiles(profiles_norm)
variance_norm = np.mean(std_norm**2) if std_norm is not None else np.inf

print(f"\n📊 Estadísticas del colapso:")
print(f"   Sin normalización: varianza media = {variance_raw:.6f}")
print(f"   Con normalización: varianza media = {variance_norm:.6f}")

# Elegir mejor normalización
use_norm = variance_norm < variance_raw
if use_norm:
    mean_U = mean_norm
    std_U = std_norm
    best_norm = "normalized amplitude"
else:
    mean_U = mean_raw
    std_U = std_raw
    best_norm = "raw amplitude"

print(f"   Mejor normalización: {best_norm}")

# ==================== AJUSTE UNIVERSAL ====================
fits = {}
r2_scores = {}

x_fit = common_grid[5:-5]
y_fit = mean_U[5:-5] if mean_U is not None else None

if y_fit is not None and len(y_fit) > 5:
    # Lineal
    try:
        popt_lin, _ = curve_fit(linear_fit, x_fit, y_fit)
        y_pred = linear_fit(x_fit, *popt_lin)
        ss_res = np.sum((y_fit - y_pred)**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
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
else:
    best_fit = "none"
    best_popt = []
    best_r2 = 0

# ==================== GUARDAR DATOS ====================
with open(POTENTIAL_DATA_FILE, 'w') as f:
    f.write("# x mean_U std_U normalization\n")
    f.write(f"# normalization: {best_norm}\n")
    if mean_U is not None:
        for i in range(len(common_grid)):
            f.write(f"{common_grid[i]:.6f} {mean_U[i]:.6f} {std_U[i]:.6f}\n")

with open(POTENTIAL_FIT_FILE, 'w') as f:
    f.write("# Transport potential collapse analysis\n")
    f.write(f"# Number of graphs: {len(profiles_raw)}\n")
    f.write(f"# Best normalization: {best_norm}\n")
    f.write(f"# Raw variance: {variance_raw:.6f}\n")
    f.write(f"# Normalized variance: {variance_norm:.6f}\n\n")
    f.write(f"Best fit: {best_fit.upper()}\n")
    f.write(f"R² = {best_r2:.4f}\n")
    f.write(f"Parameters: {list(best_popt)}\n")

print(f"💾 Datos guardados en {POTENTIAL_DATA_FILE} y {POTENTIAL_FIT_FILE}")

# ==================== FIGURAS ====================
# Figura 14: Perfiles sin normalizar
fig, ax = plt.subplots(figsize=(10, 8))
for p in profiles_raw[:10]:
    ax.plot(p['x'], p['U'], 'gray', alpha=0.3, linewidth=0.8)
if mean_raw is not None:
    ax.plot(common_grid, mean_raw, 'b-', linewidth=2, label='Media')
    ax.fill_between(common_grid, mean_raw - std_raw, mean_raw + std_raw,
                     color='blue', alpha=0.2, label='±1σ')
ax.set_xlabel('Radio normalizado $r / r_{max}$')
ax.set_ylabel('Potencial efectivo $U_{eff}(r)$')
ax.set_title('Potencial de transporte (sin normalizar)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('fig_transport_potential_collapse_raw.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_transport_potential_collapse_raw.png', dpi=300, bbox_inches='tight')
print("✅ Figura 14 guardada: fig_transport_potential_collapse_raw.pdf/png")

# Figura 15: Perfiles normalizados
fig, ax = plt.subplots(figsize=(10, 8))
for p in profiles_norm[:10]:
    ax.plot(p['x'], p['U'], 'gray', alpha=0.3, linewidth=0.8)
if mean_norm is not None:
    ax.plot(common_grid, mean_norm, 'b-', linewidth=2, label='Media')
    ax.fill_between(common_grid, mean_norm - std_norm, mean_norm + std_norm,
                     color='blue', alpha=0.2, label='±1σ')
ax.set_xlabel('Radio normalizado $r / r_{max}$')
ax.set_ylabel('Potencial normalizado $\\tilde U(r)$')
ax.set_title('Potencial de transporte (normalizado)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('fig_transport_potential_collapse_normalized.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_transport_potential_collapse_normalized.png', dpi=300, bbox_inches='tight')
print("✅ Figura 15 guardada: fig_transport_potential_collapse_normalized.pdf/png")

# Figura 16: Ajuste universal
if best_fit != "none" and mean_U is not None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.errorbar(common_grid, mean_U, yerr=std_U, fmt='o', markersize=3,
                capsize=2, color='blue', alpha=0.6, label='Datos colapsados')
    x_plot = np.linspace(0, 1, 200)
    y_plot = best_func(x_plot, *best_popt)
    ax.plot(x_plot, y_plot, 'r-', linewidth=2,
            label=f'Ajuste {best_fit.upper()}: $R^2 = {best_r2:.4f}$')
    ax.set_xlabel('Radio normalizado $r / r_{max}$')
    ax.set_ylabel('Potencial de transporte')
    ax.set_title('Ajuste universal del potencial de transporte')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig_transport_potential_universal_fit.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig_transport_potential_universal_fit.png', dpi=300, bbox_inches='tight')
    print("✅ Figura 16 guardada: fig_transport_potential_universal_fit.pdf/png")

# ==================== RESUMEN ====================
print("\n" + "="*70)
print(" RESUMEN COLAPSO DEL POTENCIAL DE TRANSPORTE")
print("="*70)
print(f"Grafos analizados: {len(profiles_raw)}")
print(f"Mejor normalización: {best_norm}")
print(f"Varianza media (mejor): {variance_norm if use_norm else variance_raw:.6f}")
if best_fit != "none":
    print(f"Mejor ajuste: {best_fit.upper()} (R² = {best_r2:.4f})")
else:
    print("Mejor ajuste: no disponible")

if (variance_norm if use_norm else variance_raw) < 0.1 and best_r2 > 0.7:
    print("\n✅ GOOD UNIVERSAL TRANSPORT PROFILE")
    print("   El potencial de transporte colapsa a una forma universal")
elif (variance_norm if use_norm else variance_raw) < 0.3 and best_r2 > 0.5:
    print("\n⚠️ PARTIAL TRANSPORT UNIVERSALITY")
    print("   Hay tendencia a universalidad pero con dispersión")
else:
    print("\n❌ NO CONVINCING TRANSPORT UNIVERSALITY")
    print("   El potencial de transporte no muestra universalidad")

print("\n✅ Script 9 completado.")