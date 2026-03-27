# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 23:57:25 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
effective_potential_analysis.py

Script 5 (adicional) – Análisis de potencial efectivo a partir de tiempos de escape.
- Calcula con mayor resolución los tiempos de escape desde cada capa radial.
- Ajusta modelos de potencial (lineal, cuadrático, logarítmico, 1/r, Schwarzschild, hiperbólico).
- Identifica la forma funcional que mejor describe la barrera efectiva.
- Permite interpretar la red en términos de potencial gravitacional análogo.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.optimize import curve_fit
from scipy.stats import linregress
import glob
import os
import re
import random
import time
import warnings
from collections import defaultdict
from sklearn.metrics import r2_score

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
MAX_GRAPHS = 8   # Suficiente para análisis detallado
SAMPLE_SIZE = 800
N_WALKERS = 500          # Mayor número para mejor estadística
MAX_STEPS = 1000         # Más pasos para evitar truncamiento
N_ESCAPE_SAMPLES = 10    # Múltiples realizaciones por capa para promediar

# Archivos de salida
POTENTIAL_DATA_FILE = "effective_potential_data.dat"
POTENTIAL_FITS_FILE = "potential_fits.dat"
BEST_MODEL_FILE = "best_potential_model.txt"

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

# ==================== TIEMPOS DE ESCAPE (ALTA RESOLUCIÓN) ====================
def escape_time_from_layer(G, center, start_layer, target_layer,
                           n_walkers=N_WALKERS, max_steps=MAX_STEPS,
                           n_samples=N_ESCAPE_SAMPLES):
    """
    Simula tiempo medio para alcanzar una capa externa desde una capa interna.
    Promedia sobre múltiples muestras para reducir ruido.
    """
    distances = nx.single_source_shortest_path_length(G, center)
    start_nodes = [n for n, d in distances.items() if d == start_layer]
    if not start_nodes:
        return None
    
    target_nodes = set([n for n, d in distances.items() if d >= target_layer])
    
    all_escape_times = []
    for _ in range(n_samples):
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
            all_escape_times.append(np.mean(escape_times))
    
    if all_escape_times:
        return np.mean(all_escape_times), np.std(all_escape_times)
    else:
        return None, None

def compute_escape_profile(G, center, max_r, n_walkers=N_WALKERS, n_samples=N_ESCAPE_SAMPLES):
    """Calcula tiempos de escape desde cada capa hasta la capa exterior."""
    escape_means = []
    escape_stds = []
    for r in range(max_r):
        t, std = escape_time_from_layer(G, center, r, max_r, n_walkers, MAX_STEPS, n_samples)
        if t is not None:
            escape_means.append(t)
            escape_stds.append(std)
        else:
            escape_means.append(np.nan)
            escape_stds.append(np.nan)
    return np.array(escape_means), np.array(escape_stds)

# ==================== MODELOS DE POTENCIAL ====================
def model_linear(r, a, b):
    """Potencial lineal: log T = a + b * r"""
    return a + b * r

def model_quadratic(r, a, b):
    """Potencial cuadrático: log T = a + b * r^2"""
    return a + b * r**2

def model_log(r, a, b):
    """Potencial logarítmico: log T = a + b * log(r)"""
    return a + b * np.log(r + 1e-6)

def model_inv(r, a, b):
    """Potencial 1/r: log T = a + b / r"""
    return a + b / (r + 1e-6)

def model_schwarzschild(r, a, b, rs):
    """Potencial tipo Schwarzschild: log T = a + b * log(1 - rs/r)"""
    # Solo válido para r > rs
    if r <= rs:
        return np.inf
    return a + b * np.log(1 - rs/(r + 1e-6))

def model_hyperbolic(r, a, b, c):
    """Potencial hiperbólico: log T = a + b*log(r) + c*r"""
    return a + b * np.log(r + 1e-6) + c * r

# ==================== PROCESAMIENTO PRINCIPAL ====================
print("="*70)
print(" SCRIPT 5: ANÁLISIS DE POTENCIAL EFECTIVO")
print("="*70)

# Buscar y seleccionar grafos
graph_files = find_graph_files()
selected = get_large_graphs(graph_files)
print(f"🎯 Seleccionados {len(selected)} grafos con N ≥ {MIN_NODES}")

all_escape_data = []

print("\n🔬 Calculando tiempos de escape (alta resolución)...")
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
        
        center, _ = approximate_barycentric_center(G)
        distances = nx.single_source_shortest_path_length(G, center)
        max_r = max(distances.values())
        
        # Perfil de escape
        escape_mean, escape_std = compute_escape_profile(G, center, max_r,
                                                         n_walkers=N_WALKERS,
                                                         n_samples=N_ESCAPE_SAMPLES)
        
        # Filtrar valores válidos
        valid = ~np.isnan(escape_mean)
        r_vals = np.arange(max_r)[valid]
        logT = np.log(escape_mean[valid])
        logT_std = escape_std[valid] / escape_mean[valid]  # error relativo en log
        
        all_escape_data.append({
            'N': N,
            'r': r_vals,
            'logT': logT,
            'logT_std': logT_std,
            'max_r': max_r
        })
        print(f"  -> max_r={max_r}, n_valid={len(r_vals)}")
        
    except Exception as e:
        print(f"  -> ERROR: {e}")
        continue

elapsed = time.time() - start_time
print(f"\n✅ Procesados {len(all_escape_data)} grafos en {elapsed:.1f} segundos")

# ==================== AJUSTE DE MODELOS ====================
print("\n" + "="*70)
print(" AJUSTE DE MODELOS DE POTENCIAL")
print("="*70)

model_functions = {
    'Linear': model_linear,
    'Quadratic': model_quadratic,
    'Logarithmic': model_log,
    '1/r': model_inv,
    'Hyperbolic': model_hyperbolic
}

results = []

for data in all_escape_data:
    N = data['N']
    r = data['r']
    logT = data['logT']
    logT_std = data['logT_std']
    
    print(f"\n📊 Grafo N={N}")
    best_r2 = -np.inf
    best_model = None
    best_params = None
    best_pred = None
    
    for name, func in model_functions.items():
        try:
            if name == 'Hyperbolic':
                popt, pcov = curve_fit(func, r, logT, p0=[0, 1, 0.1], sigma=logT_std, maxfev=5000)
                pred = func(r, *popt)
            elif name == '1/r':
                popt, pcov = curve_fit(func, r, logT, p0=[0, 1], sigma=logT_std)
                pred = func(r, *popt)
            else:
                popt, pcov = curve_fit(func, r, logT, p0=[0, 1], sigma=logT_std)
                pred = func(r, *popt)
            r2 = r2_score(logT, pred)
            print(f"   {name:12s}: R² = {r2:.4f}")
            if r2 > best_r2:
                best_r2 = r2
                best_model = name
                best_params = popt
                best_pred = pred
        except Exception as e:
            print(f"   {name:12s}: ajuste fallido")
            continue
    
    results.append({
        'N': N,
        'best_model': best_model,
        'best_r2': best_r2,
        'best_params': best_params,
        'r': r,
        'logT': logT,
        'logT_std': logT_std,
        'best_pred': best_pred
    })
    print(f"\n   🏆 Mejor modelo: {best_model} (R² = {best_r2:.4f})")

# ==================== GUARDAR RESULTADOS ====================
with open(POTENTIAL_DATA_FILE, 'w') as f:
    f.write("# N r logT logT_std\n")
    for d in all_escape_data:
        for i in range(len(d['r'])):
            f.write(f"{d['N']} {d['r'][i]} {d['logT'][i]} {d['logT_std'][i]}\n")

with open(POTENTIAL_FITS_FILE, 'w') as f:
    f.write("# N best_model best_r2 params\n")
    for r in results:
        f.write(f"{r['N']} {r['best_model']} {r['best_r2']:.4f} {r['best_params']}\n")

# Resumen de modelos ganadores
model_counts = defaultdict(int)
for r in results:
    model_counts[r['best_model']] += 1

with open(BEST_MODEL_FILE, 'w') as f:
    f.write("# Modelo predominante:\n")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        f.write(f"{model}: {count}/{len(results)} ({100*count/len(results):.1f}%)\n")
    f.write("\n# Mejor R² medio por modelo:\n")
    for model in model_functions.keys():
        r2s = [r['best_r2'] for r in results if r['best_model'] == model]
        if r2s:
            f.write(f"{model}: media={np.mean(r2s):.4f} ± {np.std(r2s):.4f}\n")

print(f"\n💾 Datos guardados en {POTENTIAL_DATA_FILE}, {POTENTIAL_FITS_FILE}, {BEST_MODEL_FILE}")

# ==================== FIGURAS ====================
# Figura 10: Ajustes de potencial para cada grafo
n_plots = len(results)
fig, axes = plt.subplots(2, (n_plots+1)//2, figsize=(14, 10))
axes = axes.flatten() if n_plots > 1 else [axes]

for i, r in enumerate(results):
    ax = axes[i]
    ax.errorbar(r['r'], r['logT'], yerr=r['logT_std'], fmt='o', capsize=3,
                markersize=4, label='Datos')
    ax.plot(r['r'], r['best_pred'], 'r-', linewidth=2,
            label=f'{r["best_model"]} (R²={r["best_r2"]:.3f})')
    ax.set_xlabel('Radio r')
    ax.set_ylabel('log T_escape')
    ax.set_title(f'N={r["N"]}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# Eliminar ejes vacíos si los hay
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Ajuste de potencial efectivo', fontsize=14)
plt.tight_layout()
plt.savefig('fig_potential_fits.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_potential_fits.png', dpi=300, bbox_inches='tight')
print("✅ Figura 10 guardada: fig_potential_fits.pdf/png")

# Figura 11: Comparación de modelos (diagrama de barras)
plt.figure(figsize=(10, 6))
models = list(model_counts.keys())
counts = [model_counts[m] for m in models]
bars = plt.bar(models, counts, color='skyblue', edgecolor='black')
plt.xlabel('Modelo de potencial')
plt.ylabel('Número de grafos')
plt.title('Modelo predominante por grafo')
plt.xticks(rotation=45)
for bar, cnt in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{cnt}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('fig_potential_model_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_potential_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Figura 11 guardada: fig_potential_model_comparison.pdf/png")

# ==================== RESUMEN ====================
print("\n" + "="*70)
print(" RESUMEN DE POTENCIAL EFECTIVO")
print("="*70)
print(f"Grafos analizados: {len(results)}")

for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
    print(f"   {model}: {count}/{len(results)} ({100*count/len(results):.1f}%)")

print("\n📈 Mejor R² por modelo:")
for model in model_functions.keys():
    r2s = [r['best_r2'] for r in results if r['best_model'] == model]
    if r2s:
        print(f"   {model}: media={np.mean(r2s):.4f} ± {np.std(r2s):.4f}")

# Conclusión
best_global_model = max(model_counts, key=model_counts.get)
print(f"\n🏆 Modelo de potencial dominante: {best_global_model}")

if best_global_model == 'Quadratic':
    print("   → Potencial cuadrático: comportamiento tipo oscilador armónico")
elif best_global_model == 'Linear':
    print("   → Potencial lineal: fuerza constante, típico de campo eléctrico uniforme")
elif best_global_model == 'Logarithmic':
    print("   → Potencial logarítmico: característico de geometría hiperbólica")
elif best_global_model == '1/r':
    print("   → Potencial 1/r: análogo a campo gravitatorio newtoniano (estrella, agujero negro)") 
elif best_global_model == 'Hyperbolic':
    print("   → Potencial hiperbólico: combinación logarítmica + lineal, típico de espacios con curvatura negativa")

print("\n✅ Script 5 completado.")