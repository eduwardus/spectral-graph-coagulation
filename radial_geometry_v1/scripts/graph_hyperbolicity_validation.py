#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_hyperbolicity_validation.py

Script 7: Validación de hiperbolicidad con Gromov δ (VERSIÓN CORREGIDA).
- Calcula la δ-hiperbolicidad de Gromov para grafos grandes.
- Usa muestreo estratificado de cuádruplas para mejor cobertura.
- Implementa la fórmula correcta: δ = max( (S3 - S2)/2 ) sobre cuádruplas.
- Compara con referencias: árbol, lattice 3D, grafo hiperbólico sintético.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import glob
import os
import re
import random
import time
import heapq
from collections import defaultdict
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
MAX_GRAPHS = 10  # Reducido para mantener tiempo razonable
N_SAMPLES = 5000  # Número de cuádruplas aleatorias
N_CANDIDATE_NODES = 50  # Nodos candidatos para muestreo de pares extremos
N_PAIR_SAMPLES = 500  # Número de pares extremos a muestrear

# Archivos de salida
HYPERBOLICITY_DATA_FILE = "hyperbolicity_data.dat"
COMPARISON_FILE = "hyperbolicity_comparison.dat"

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

# ==================== CÁLCULO DE DISTANCIAS EFICIENTE ====================
class DistanceCache:
    """Caché para distancias entre nodos."""
    def __init__(self, G):
        self.G = G
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, a, b):
        if a == b:
            return 0
        key = (min(a, b), max(a, b))
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        d = nx.shortest_path_length(self.G, a, b)
        self.cache[key] = d
        return d
    
    def stats(self):
        return self.hits, self.misses

def compute_hyperbolicity_fast(G, n_samples=5000, n_candidates=50, n_pairs=500):
    """
    Calcula δ-hiperbolicidad de Gromov con muestreo inteligente.
    
    Estrategia:
    1. Seleccionar nodos candidatos para pares extremos (los de mayor grado y aleatorios)
    2. Muestrear pares extremos y calcular distancias
    3. Para cada par extremo, muestrear otros nodos para formar cuádruplas
    4. Calcular δ correctamente: δ = max( (S3 - S2)/2 )
    """
    nodes = list(G.nodes())
    N = len(nodes)
    
    if N < 10:
        return 0.0, 0, 0
    
    # Precomputar grados
    degrees = dict(G.degree())
    
    # Seleccionar nodos candidatos (top por grado + aleatorios)
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    high_deg = sorted_nodes[:n_candidates//2]
    random_nodes = random.sample(nodes, min(n_candidates//2, len(nodes)))
    candidate_nodes = list(set(high_deg + random_nodes))
    
    # Para muestreo de pares extremos, necesitamos una buena cobertura del diámetro
    # Tomamos nodos aleatorios y sus nodos más lejanos
    
    # Pre-seleccionar pares extremos
    extreme_pairs = []
    for _ in range(min(n_pairs, len(candidate_nodes))):
        a = random.choice(candidate_nodes)
        # BFS rápido para encontrar nodo más lejano
        try:
            lengths = nx.single_source_shortest_path_length(G, a)
            far_node = max(lengths, key=lengths.get)
            extreme_pairs.append((a, far_node))
        except:
            continue
    
    # También añadir algunos pares aleatorios
    for _ in range(min(n_pairs // 2, 100)):
        a = random.choice(nodes)
        b = random.choice(nodes)
        extreme_pairs.append((a, b))
    
    # Eliminar duplicados
    extreme_pairs = list(set([(min(p), max(p)) for p in extreme_pairs]))
    
    # Crear caché de distancias
    cache = DistanceCache(G)
    
    delta = 0.0
    sampled = 0
    max_diameter = 0
    
    # Para cada par extremo, muestrear otros nodos
    for a, b in extreme_pairs[:n_pairs]:
        # Distancia entre los extremos
        d_ab = cache.get(a, b)
        max_diameter = max(max_diameter, d_ab)
        
        # Muestrear otros nodos
        other_nodes = random.sample(nodes, min(n_samples // len(extreme_pairs), len(nodes)))
        
        for c in other_nodes:
            if c == a or c == b:
                continue
            
            d_ac = cache.get(a, c)
            d_bc = cache.get(b, c)
            
            # Ya tenemos d_ab, ahora probar con un cuarto nodo
            for d in other_nodes:
                if d in (a, b, c):
                    continue
                
                d_ad = cache.get(a, d)
                d_bd = cache.get(b, d)
                d_cd = cache.get(c, d)
                
                # Las tres sumas de pares opuestos
                sums = [
                    d_ab + d_cd,
                    d_ac + d_bd,
                    d_ad + d_bc
                ]
                sums.sort()  # sums[0] ≤ sums[1] ≤ sums[2]
                
                # δ para esta cuádrupla
                delta_quad = (sums[2] - sums[1]) / 2.0
                delta = max(delta, delta_quad)
                sampled += 1
                
                # Si ya tenemos un delta grande, podemos parar temprano
                if delta > max_diameter:
                    break
            if delta > max_diameter:
                break
        if delta > max_diameter:
            break
    
    # Normalizar por el diámetro para obtener δ_norm
    if max_diameter > 0:
        delta_norm = delta / max_diameter
    else:
        delta_norm = 0
    
    return delta, max_diameter, delta_norm, sampled, cache.stats()

def gromov_hyperbolicity_simple(G, n_samples=10000):
    """
    Versión simple con muestreo aleatorio de cuádruplas (para referencias más pequeñas).
    """
    nodes = list(G.nodes())
    N = len(nodes)
    
    if N < 4:
        return 0, 1, 0
    
    cache = DistanceCache(G)
    delta = 0.0
    max_diameter = 0
    sampled = 0
    
    for _ in range(n_samples):
        try:
            a, b, c, d = random.sample(nodes, 4)
        except ValueError:
            break
        
        d_ab = cache.get(a, b)
        d_cd = cache.get(c, d)
        d_ac = cache.get(a, c)
        d_bd = cache.get(b, d)
        d_ad = cache.get(a, d)
        d_bc = cache.get(b, c)
        
        max_diameter = max(max_diameter, d_ab, d_cd, d_ac, d_bd, d_ad, d_bc)
        
        sums = [d_ab + d_cd, d_ac + d_bd, d_ad + d_bc]
        sums.sort()
        delta_quad = (sums[2] - sums[1]) / 2.0
        delta = max(delta, delta_quad)
        sampled += 1
    
    if max_diameter > 0:
        delta_norm = delta / max_diameter
    else:
        delta_norm = 0
    
    return delta, max_diameter, delta_norm, sampled, cache.stats()

# ==================== GENERAR REFERENCIAS ====================
def generate_reference_graphs():
    """Genera grafos de referencia para comparación."""
    references = {}
    
    # 1. Árbol regular
    print("   Generando árbol regular...")
    G_tree = nx.balanced_tree(4, 4)
    if G_tree.number_of_nodes() < 500:
        G_tree = nx.balanced_tree(5, 4)
    references['Regular Tree'] = G_tree
    
    # 2. Lattice 3D
    print("   Generando lattice 3D...")
    G_cube = nx.grid_graph(dim=[15, 15, 15])
    G_cube_simple = nx.Graph()
    for u, v in G_cube.edges():
        G_cube_simple.add_edge(u, v)
    references['3D Cubic Lattice'] = G_cube_simple
    
    # 3. Grafo hiperbólico sintético (Watts-Strogatz con clustering)
    print("   Generando grafo hiperbólico sintético...")
    G_hyp = nx.watts_strogatz_graph(2000, 8, 0.3, seed=RANDOM_SEED)
    if not nx.is_connected(G_hyp):
        largest = max(nx.connected_components(G_hyp), key=len)
        G_hyp = G_hyp.subgraph(largest).copy()
    references['Hyperbolic-like (WS)'] = G_hyp
    
    return references

# ==================== PROCESAMIENTO PRINCIPAL ====================
print("="*70)
print(" BLOQUE 7: VALIDACIÓN DE HIPERBOLICIDAD (GROMOV δ) - VERSIÓN CORREGIDA")
print("="*70)

# Buscar grafos
graph_files = find_graph_files()
print(f"📁 Total archivos .npz: {len(graph_files)}")

selected = get_large_graphs(graph_files)
print(f"🎯 Seleccionados {len(selected)} grafos con N ≥ {MIN_NODES}")
if selected:
    print(f"   Rango: {selected[0][1]} - {selected[-1][1]}")

# Analizar grafos del modelo
model_results = []

print("\n🔬 Calculando δ-hiperbolicidad para grafos del modelo...")
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
        
        # Para grafos grandes (>20000 nodos) usar menos muestras por tiempo
        if N > 20000:
            delta, diam, delta_norm, sampled, (hits, misses) = compute_hyperbolicity_fast(
                G, n_samples=3000, n_candidates=40, n_pairs=300
            )
        else:
            delta, diam, delta_norm, sampled, (hits, misses) = compute_hyperbolicity_fast(
                G, n_samples=5000, n_candidates=50, n_pairs=500
            )
        
        model_results.append({
            'N': N,
            'delta': delta,
            'diameter': diam,
            'delta_norm': delta_norm,
            'sampled': sampled,
            'cache_hits': hits,
            'cache_misses': misses
        })
        
        print(f"  -> δ={delta:.2f}, diam={diam}, δ_norm={delta_norm:.4f}, sampled={sampled}")
        
    except Exception as e:
        print(f"  -> ERROR: {e}")
        continue

elapsed = time.time() - start_time
print(f"\n✅ Procesados {len(model_results)} grafos en {elapsed:.1f} segundos")

# Analizar grafos de referencia
print("\n🔬 Generando y analizando grafos de referencia...")
references = generate_reference_graphs()
ref_results = []

for name, G in references.items():
    print(f"   {name}...", end='', flush=True)
    try:
        delta, diam, delta_norm, sampled, (hits, misses) = gromov_hyperbolicity_simple(G, n_samples=5000)
        ref_results.append({
            'name': name,
            'N': G.number_of_nodes(),
            'delta': delta,
            'diameter': diam,
            'delta_norm': delta_norm,
            'sampled': sampled
        })
        print(f"  N={G.number_of_nodes()}, δ_norm={delta_norm:.4f}")
    except Exception as e:
        print(f"  -> ERROR: {e}")
        continue

# ==================== GUARDAR DATOS ====================
with open(HYPERBOLICITY_DATA_FILE, 'w') as f:
    f.write("# N delta diameter delta_norm sampled cache_hits cache_misses\n")
    for r in model_results:
        f.write(f"{r['N']} {r['delta']:.2f} {r['diameter']} {r['delta_norm']:.6f} {r['sampled']} {r['cache_hits']} {r['cache_misses']}\n")

with open(COMPARISON_FILE, 'w') as f:
    f.write("# name N delta diameter delta_norm\n")
    for r in ref_results:
        f.write(f"{r['name']} {r['N']} {r['delta']:.2f} {r['diameter']} {r['delta_norm']:.6f}\n")

print(f"💾 Datos guardados en {HYPERBOLICITY_DATA_FILE} y {COMPARISON_FILE}")

# ==================== FIGURAS ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Figura A: δ_norm vs N
ax = axes[0, 0]
if model_results:
    Ns = [r['N'] for r in model_results]
    delta_norm = [r['delta_norm'] for r in model_results]
    ax.scatter(Ns, delta_norm, s=50, c='blue', alpha=0.7, label='Modelo')
    ax.set_xscale('log')
    ax.set_xlabel('Tamaño N')
    ax.set_ylabel('δ / diámetro')
    ax.set_title('A. Hiperbolicidad vs tamaño')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Figura B: Comparación con referencias
ax = axes[0, 1]
if ref_results:
    names = [r['name'] for r in ref_results]
    ref_delta = [r['delta_norm'] for r in ref_results]
    ax.bar(names, ref_delta, color=['green', 'orange', 'red'])
    if model_results:
        ax.axhline(y=np.mean(delta_norm), color='blue', linestyle='--', 
                   label=f'Modelo (media={np.mean(delta_norm):.3f})')
    ax.set_ylabel('δ / diámetro')
    ax.set_title('B. Comparación con referencias')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

# Figura C: Histograma
ax = axes[1, 0]
if model_results:
    ax.hist(delta_norm, bins=10, alpha=0.5, color='blue', label='Modelo')
    hyp_ref = [r for r in ref_results if 'Hyperbolic' in r['name']]
    if hyp_ref:
        ax.axvline(x=hyp_ref[0]['delta_norm'], color='red', linestyle='--', 
                   label=f'Hiperbólico ref: {hyp_ref[0]["delta_norm"]:.3f}')
    ax.set_xlabel('δ / diámetro')
    ax.set_ylabel('Frecuencia')
    ax.set_title('C. Distribución de hiperbolicidad')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Figura D: δ vs diámetro
ax = axes[1, 1]
if model_results:
    for r in model_results:
        ax.scatter(r['diameter'], r['delta'], s=50, c='blue', alpha=0.7)
for r in ref_results:
    ax.scatter(r['diameter'], r['delta'], s=80, marker='s', 
               label=r['name'], alpha=0.7)
ax.set_xlabel('Diámetro')
ax.set_ylabel('δ')
ax.set_title('D. δ vs diámetro')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('fig_hyperbolicity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_hyperbolicity.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: fig_hyperbolicity.pdf/png")

# ==================== RESUMEN ====================
print("\n" + "="*70)
print(" RESUMEN HIPERBOLICIDAD")
print("="*70)
if model_results:
    print(f"Grafos analizados: {len(model_results)}")
    print(f"δ_norm medio: {np.mean(delta_norm):.4f} ± {np.std(delta_norm):.4f}")
    print(f"δ_norm mediana: {np.median(delta_norm):.4f}")
    print(f"δ_norm máximo: {np.max(delta_norm):.4f}")
    print(f"δ_norm mínimo: {np.min(delta_norm):.4f}")

print("\n📊 Comparación con referencias:")
for r in ref_results:
    print(f"   {r['name']}: δ_norm = {r['delta_norm']:.4f} (N={r['N']})")

# Interpretación
if model_results:
    mean_delta_norm = np.mean(delta_norm)
    if mean_delta_norm < 0.1:
        print("\n✅ HIPERBOLICIDAD FUERTE (δ_norm < 0.1)")
        print("   La red es casi un árbol en términos de Gromov")
    elif mean_delta_norm < 0.3:
        print("\n✅ HIPERBOLICIDAD MODERADA")
        print("   Consistente con geometría hiperbólica efectiva")
    elif mean_delta_norm < 0.5:
        print("\n⚠️ HIPERBOLICIDAD DÉBIL")
        print("   La red es menos hiperbólica que una geometría típica")
    else:
        print("\n❌ NO HIPERBÓLICA")
        print("   La red no muestra hiperbolicidad efectiva")

print("\n✅ Script 7 completado.")