#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectral_dimension_finite_size_scaling.py (VERSIÓN SIMPLIFICADA Y ROBUSTA)

Análisis de finite-size scaling para la dimensión espectral.
- Calcula probabilidad de retorno P(t) mediante random walks Monte Carlo
- Estima dimensión espectral ajustando P(t) ~ t^{-d_s/2} en la región de meseta
- Detecta meseta automáticamente
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.optimize import curve_fit
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

GRAPH_DIRS = [
    "soup_simulation_phase_transition_v20",
    "soup_simulation_phase_transition_v20/snapshots"
]

MIN_NODES = 500
MAX_GRAPHS_PER_BIN = 8
N_WALKS = 500
N_SEEDS = 20
MAX_T_STEPS = 200
T_MIN = 5

# Archivos de salida
RETURN_PROB_DATA = "return_probability_fss_data.dat"
PLATEAU_DATA = "spectral_dimension_plateau_data.dat"

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

def get_large_graphs_by_bin(files, min_nodes=MIN_NODES, max_per_bin=MAX_GRAPHS_PER_BIN):
    """Obtiene grafos agrupados por bins logarítmicos."""
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
    
    # Bins logarítmicos
    logNs = np.log([c[1] for c in candidates])
    if len(logNs) < 2:
        return []
    
    bins = np.linspace(logNs.min(), logNs.max(), 5)
    selected_by_bin = []
    
    for i in range(len(bins)-1):
        bin_cands = [c for c in candidates if bins[i] <= np.log(c[1]) < bins[i+1]]
        if bin_cands:
            # Tomar muestra representativa
            step = max(1, len(bin_cands) // max_per_bin)
            selected = bin_cands[::step][:max_per_bin]
            selected_by_bin.append((np.exp(bins[i]), np.exp(bins[i+1]), selected))
    
    return selected_by_bin

def get_largest_component(G):
    if nx.is_connected(G):
        return G
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()

def compute_return_probability(G, t_max, n_seeds=N_SEEDS, n_walks=N_WALKS):
    """Estima P(t) mediante simulación Monte Carlo."""
    nodes = list(G.nodes())
    if len(nodes) < n_seeds:
        n_seeds = len(nodes)
    seeds = random.sample(nodes, n_seeds)
    
    P_t = np.zeros(t_max + 1)
    
    for seed in seeds:
        for _ in range(n_walks):
            pos = seed
            P_t[0] += 1
            for t in range(1, t_max + 1):
                nb = list(G.neighbors(pos))
                if nb:
                    pos = random.choice(nb)
                if pos == seed:
                    P_t[t] += 1
    
    P_t = P_t / (n_seeds * n_walks)
    return P_t

def estimate_spectral_dimension(P, t_max, t_min=T_MIN):
    """Estima d_s ajustando P(t) ~ t^{-d_s/2} en el rango donde la curva es suave."""
    t = np.arange(1, t_max + 1)
    
    # Filtrar valores donde P es demasiado pequeño
    mask = (P[1:] > 1e-6) & (t >= t_min)
    if np.sum(mask) < 10:
        return None, None, None
    
    t_fit = t[mask]
    P_fit = P[1:][mask]
    
    # Buscar el mejor rango de ajuste
    best_ds = None
    best_r2 = -1
    best_range = None
    
    # Probar diferentes ventanas de ajuste
    for start in range(len(t_fit)//4):
        for end in range(len(t_fit)//2, len(t_fit)):
            if end - start < 10:
                continue
            
            t_win = t_fit[start:end]
            P_win = P_fit[start:end]
            
            logt = np.log(t_win)
            logP = np.log(P_win)
            
            try:
                slope, intercept, r_value, _, _ = np.polyfit(logt, logP, 1, full=False)
                r2 = r_value**2
                ds = -2 * slope
                
                # Aceptar solo valores físicos
                if 0.5 < ds < 5.0 and r2 > 0.7:
                    if r2 > best_r2:
                        best_ds = ds
                        best_r2 = r2
                        best_range = (t_win[0], t_win[-1])
            except:
                continue
    
    return best_ds, best_r2, best_range

def power_law(t, a, ds):
    """P(t) = a * t^{-ds/2}"""
    return a * t**(-ds/2)

def estimate_with_fit(P, t_max, t_min=T_MIN):
    """Estima d_s mediante ajuste no lineal."""
    t = np.arange(1, t_max + 1)
    mask = (P[1:] > 1e-6) & (t >= t_min)
    
    if np.sum(mask) < 10:
        return None, None
    
    t_fit = t[mask]
    P_fit = P[1:][mask]
    
    try:
        popt, pcov = curve_fit(power_law, t_fit, P_fit, p0=[P_fit[0], 2.0])
        ds = popt[1]
        r2 = 1 - np.sum((P_fit - power_law(t_fit, *popt))**2) / np.sum((P_fit - np.mean(P_fit))**2)
        return ds, r2
    except:
        return None, None

# ==================== PROCESAMIENTO PRINCIPAL ====================
print("="*70)
print(" SCRIPT 1: FINITE-SIZE SCALING DE LA DIMENSIÓN ESPECTRAL (SIMPLIFICADO)")
print("="*70)

graph_files = find_graph_files()
print(f"📁 Total archivos .npz: {len(graph_files)}")

# Obtener grafos por bins
binned_graphs = get_large_graphs_by_bin(graph_files)
print(f"📊 {len(binned_graphs)} bins de tamaño")

all_results = []  # (N, ds, r2, t_range)
all_P_curves = []  # (N, t, P)

print("\n🔬 Procesando grafos...")
start_time = time.time()

for bin_idx, (low, high, graphs) in enumerate(binned_graphs):
    print(f"\n📊 Bin {bin_idx+1}: N ∈ [{low:.0f}, {high:.0f}] ({len(graphs)} grafos)")
    
    for fname, N in graphs:
        print(f"      N={N}...", end='', flush=True)
        try:
            A = load_npz(fname)
            G = nx.from_scipy_sparse_array(A)
            G = get_largest_component(G)
            N_actual = G.number_of_nodes()
            if N_actual < MIN_NODES:
                print("  -> omitido")
                continue
            
            # t_max proporcional al log del tamaño
            t_max = min(MAX_T_STEPS, max(40, int(8 * np.log2(N_actual))))
            
            # Calcular P(t)
            P = compute_return_probability(G, t_max, n_seeds=min(15, N_actual//20), n_walks=200)
            
            # Estimar d_s
            ds, r2, t_range = estimate_spectral_dimension(P, t_max, t_min=T_MIN)
            
            if ds is None:
                print("  -> no se pudo estimar")
                continue
            
            print(f"  -> ds={ds:.3f}, R²={r2:.3f}, t∈[{t_range[0]:.0f},{t_range[1]:.0f}]")
            
            all_results.append({
                'N': N_actual,
                'ds': ds,
                'r2': r2,
                't_min': t_range[0] if t_range else None,
                't_max': t_range[1] if t_range else None
            })
            
            all_P_curves.append({
                'N': N_actual,
                't': np.arange(1, t_max+1),
                'P': P[1:],
                'ds': ds
            })
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

elapsed = time.time() - start_time
print(f"\n✅ Procesados {len(all_results)} grafos válidos en {elapsed:.1f} segundos")

if len(all_results) == 0:
    print("❌ No se obtuvieron resultados válidos")
    exit()

# ==================== ANÁLISIS POR BINS ====================
print("\n" + "="*70)
print(" ANÁLISIS POR BINS DE TAMAÑO")
print("="*70)

# Reagrupar resultados por tamaño
size_groups = []
for low, high in [(500, 1000), (1000, 3000), (3000, 10000), (10000, 30000), (30000, 1e9)]:
    group = [r for r in all_results if low <= r['N'] < high]
    if group:
        ds_vals = [r['ds'] for r in group]
        size_groups.append({
            'range': f"{low}-{high}",
            'mean': np.mean(ds_vals),
            'std': np.std(ds_vals),
            'n': len(group),
            'ds_vals': ds_vals
        })
        print(f"\nBin {low}-{high}:")
        print(f"   n={len(group)}, ds={np.mean(ds_vals):.3f}±{np.std(ds_vals):.3f}")

# ==================== FIGURAS ====================
# Figura 1: Probabilidad de retorno
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0, 1, min(15, len(all_P_curves))))
for i, pcurve in enumerate(all_P_curves[:15]):
    ax.loglog(pcurve['t'], pcurve['P'], color=colors[i], alpha=0.7, linewidth=1, label=f"N={pcurve['N']}")
ax.set_xlabel('Tiempo t')
ax.set_ylabel('Probabilidad de retorno P(t)')
ax.set_title('Probabilidad de retorno por tamaño')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower left', fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig('fig_return_probability_by_size.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_return_probability_by_size.png', dpi=300, bbox_inches='tight')
print("✅ Figura 1: fig_return_probability_by_size.pdf/png")

# Figura 2: d_s vs N
fig, ax = plt.subplots(figsize=(10, 8))
Ns = [r['N'] for r in all_results]
ds_vals = [r['ds'] for r in all_results]
ax.scatter(Ns, ds_vals, alpha=0.6, s=40, c='blue')
ax.axhline(y=3.0, color='red', linestyle='--', label='d_s = 3')
ax.set_xscale('log')
ax.set_xlabel('Tamaño del grafo N')
ax.set_ylabel('Dimensión espectral d_s')
ax.set_title('Dimensión espectral vs tamaño')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('fig_ds_plateau_vs_size.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_ds_plateau_vs_size.png', dpi=300, bbox_inches='tight')
print("✅ Figura 2: fig_ds_plateau_vs_size.pdf/png")

# Figura 3: Calidad del ajuste
fig, ax = plt.subplots(figsize=(10, 8))
r2_vals = [r['r2'] for r in all_results]
ax.scatter(Ns, r2_vals, alpha=0.6, s=40, c='green')
ax.axhline(y=0.8, color='red', linestyle='--', label='R² = 0.8')
ax.set_xscale('log')
ax.set_xlabel('Tamaño del grafo N')
ax.set_ylabel('Coeficiente de determinación R²')
ax.set_title('Calidad del ajuste espectral')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('fig_running_ds_by_size.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_running_ds_by_size.png', dpi=300, bbox_inches='tight')
print("✅ Figura 3: fig_running_ds_by_size.pdf/png")

# Figura 4: Colapso de curvas
fig, ax = plt.subplots(figsize=(10, 8))
for pcurve in all_P_curves:
    if pcurve['ds'] > 0:
        t_scaled = pcurve['t'] / pcurve['t'][-1]
        P_scaled = pcurve['P'] * (pcurve['t'] ** (pcurve['ds']/2))
        ax.plot(t_scaled, P_scaled, alpha=0.5, linewidth=0.8)
ax.set_xscale('log')
ax.set_xlabel('t / t_max')
ax.set_ylabel('P(t) * t^{d_s/2}')
ax.set_title('Colapso de curvas (con d_s individual)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_return_probability_collapse.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_return_probability_collapse.png', dpi=300, bbox_inches='tight')
print("✅ Figura 4: fig_return_probability_collapse.pdf/png")

# ==================== GUARDAR DATOS ====================
with open(RETURN_PROB_DATA, 'w') as f:
    f.write("# N t P(t)\n")
    for pcurve in all_P_curves:
        for i in range(len(pcurve['t'])):
            f.write(f"{pcurve['N']} {pcurve['t'][i]} {pcurve['P'][i]:.8e}\n")

with open(PLATEAU_DATA, 'w') as f:
    f.write("# N ds r2 t_min t_max\n")
    for r in all_results:
        f.write(f"{r['N']} {r['ds']:.4f} {r['r2']:.4f} {r['t_min'] if r['t_min'] else -1} {r['t_max'] if r['t_max'] else -1}\n")

print(f"💾 Datos guardados en {RETURN_PROB_DATA} y {PLATEAU_DATA}")

# ==================== RESUMEN FINAL ====================
print("\n" + "="*70)
print(" RESUMEN FINAL")
print("="*70)
print(f"Grafos analizados: {len(all_results)}")
print(f"Rango de tamaños: {min(Ns):.0f} - {max(Ns):.0f}")
print(f"d_s medio: {np.mean(ds_vals):.3f} ± {np.std(ds_vals):.3f}")

if size_groups:
    print("\n📊 d_s por bin de tamaño:")
    for g in size_groups:
        print(f"   {g['range']}: {g['mean']:.3f} ± {g['std']:.3f} (n={g['n']})")
    
    # Verificar estabilización
    large_bins = [g for g in size_groups if '10000' in g['range'] or '30000' in g['range']]
    if large_bins:
        large_vals = [g['mean'] for g in large_bins]
        if len(large_vals) >= 2 and abs(large_vals[-1] - large_vals[0]) < 0.3:
            print("\n✅ ESTABILIZACIÓN EN GRAFOS GRANDES")
            print(f"   d_s ≈ {np.mean(large_vals):.3f} ± {np.std(large_vals):.3f}")
        else:
            print("\n⚠️ TENDENCIA AÚN PRESENTE")

print("\n✅ Script 1 completado.")