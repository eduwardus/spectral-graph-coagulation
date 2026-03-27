# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 23:49:48 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reference_geometry_comparison.py

Script 4 del análisis de geometría radial efectiva.
- Genera o carga grafos de referencia con geometrías conocidas:
  * Lattice cúbica 3D
  * Árbol regular (grafo tipo estrella o árbol binario)
  * Grafo hiperbólico aproximado (modelo de red hiperbólica)
  * Modelo radial sintético con control de conectividad
- Compara los perfiles radiales del modelo con estas referencias.
- Calcula distancias entre perfiles y encuentra la mejor coincidencia.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
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

# Directorios
GRAPH_DIRS = [
    "soup_simulation_phase_transition_v20",
    "soup_simulation_phase_transition_v20/snapshots"
]

MIN_NODES = 5000
MAX_GRAPHS = 8  # Suficiente para comparación
SAMPLE_SIZE = 800

# Archivos de salida
COMPARISON_FILE = "reference_geometry_comparison.dat"
BEST_MATCH_FILE = "best_match_summary.txt"

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

def get_radial_profile(G, center, max_r=None):
    """Devuelve A(r) y V(r) para un grafo dado un centro."""
    distances = nx.single_source_shortest_path_length(G, center)
    max_dist = max(distances.values())
    if max_r is not None:
        max_dist = min(max_dist, max_r)
    
    A = np.zeros(max_dist+1, dtype=int)
    V = np.zeros(max_dist+1, dtype=int)
    for d in distances.values():
        if d <= max_dist:
            A[d] += 1
    V[0] = A[0]
    for r in range(1, max_dist+1):
        V[r] = V[r-1] + A[r]
    return np.arange(max_dist+1), A, V

# ==================== GENERACIÓN DE REFERENCIAS ====================
def generate_3d_lattice(n_side=20):
    """Genera un lattice cúbico 3D con aproximadamente N = n_side^3 nodos."""
    G = nx.grid_graph(dim=[n_side, n_side, n_side])
    # Convertir a grafo simple (sin atributos de coordenadas)
    G_simple = nx.Graph()
    for u, v in G.edges():
        G_simple.add_edge(u, v)
    return G_simple

def generate_regular_tree(branching=3, depth=6):
    """Genera un árbol regular con branching y depth."""
    # Usamos el grafo de árbol balanceado
    G = nx.balanced_tree(branching, depth)
    return G

def generate_hyperbolic_graph(N, k=3, alpha=0.5):
    """
    Genera un grafo hiperbólico aproximado usando el modelo de red hiperbólica.
    Simulación simplificada: genera nodos en un disco hiperbólico y conecta según
    la distancia hiperbólica.
    """
    # Para evitar complejidad, generamos un grafo small-world con clustering
    # como aproximación a geometría hiperbólica
    G = nx.watts_strogatz_graph(N, k, alpha, seed=RANDOM_SEED)
    # Asegurar conectividad
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    return G

def generate_radial_synthetic_model(center_degree=20, decay=0.8, n_layers=15):
    """
    Modelo radial sintético con control de conectividad.
    - Capa 0: nodo central con grado center_degree
    - Cada capa r tiene conectividad radial decreciente
    """
    G = nx.Graph()
    nodes_per_layer = [1]  # capa 0: centro
    
    # Número de nodos por capa (decreciente para simular curvatura negativa)
    for r in range(1, n_layers+1):
        n_nodes = int(center_degree * (decay ** r) * 10)
        n_nodes = max(1, n_nodes)
        nodes_per_layer.append(n_nodes)
    
    # Crear nodos
    node_id = 0
    layer_nodes = []
    for r, n in enumerate(nodes_per_layer):
        layer = []
        for _ in range(n):
            G.add_node(node_id, layer=r)
            layer.append(node_id)
            node_id += 1
        layer_nodes.append(layer)
    
    # Conectar radialmente
    for r in range(1, len(layer_nodes)):
        for node in layer_nodes[r]:
            # Conectar a nodos de la capa anterior
            parents = random.sample(layer_nodes[r-1], min(2, len(layer_nodes[r-1])))
            for p in parents:
                G.add_edge(node, p)
    
    # Añadir conexiones tangenciales (para crear ciclos)
    for r in range(1, len(layer_nodes)):
        # Conectar nodos dentro de la misma capa
        for i in range(len(layer_nodes[r])):
            for j in range(i+1, min(i+3, len(layer_nodes[r]))):
                if random.random() < 0.3:
                    G.add_edge(layer_nodes[r][i], layer_nodes[r][j])
    
    return G

# ==================== CARGAR O GENERAR REFERENCIAS ====================
print("="*70)
print(" BLOQUE 4: COMPARACIÓN CON GEOMETRÍAS DE REFERENCIA")
print("="*70)

# Generar o cargar referencias
references = {}

print("\n🔨 Generando geometrías de referencia...")

# 1. Lattice 3D
print("   Generando lattice 3D...")
G_cube = generate_3d_lattice(n_side=25)
center_cube = list(G_cube.nodes())[len(G_cube.nodes())//2]  # centro aproximado
r_cube, A_cube, V_cube = get_radial_profile(G_cube, center_cube)
references['3D Cubic Lattice'] = {'G': G_cube, 'center': center_cube, 'r': r_cube, 'A': A_cube, 'V': V_cube}
print(f"      Nodos: {G_cube.number_of_nodes()}")

# 2. Árbol regular
print("   Generando árbol regular...")
G_tree = generate_regular_tree(branching=3, depth=7)
center_tree = list(G_tree.nodes())[0]  # raíz
r_tree, A_tree, V_tree = get_radial_profile(G_tree, center_tree)
references['Regular Tree'] = {'G': G_tree, 'center': center_tree, 'r': r_tree, 'A': A_tree, 'V': V_tree}
print(f"      Nodos: {G_tree.number_of_nodes()}")

# 3. Grafo hiperbólico aproximado
print("   Generando grafo hiperbólico aproximado...")
G_hyp = generate_hyperbolic_graph(5000, k=5, alpha=0.3)
center_hyp, _ = approximate_barycentric_center(G_hyp)
r_hyp, A_hyp, V_hyp = get_radial_profile(G_hyp, center_hyp, max_r=15)
references['Hyperbolic-like'] = {'G': G_hyp, 'center': center_hyp, 'r': r_hyp, 'A': A_hyp, 'V': V_hyp}
print(f"      Nodos: {G_hyp.number_of_nodes()}")

# 4. Modelo radial sintético
print("   Generando modelo radial sintético...")
G_synth = generate_radial_synthetic_model(center_degree=25, decay=0.7, n_layers=12)
center_synth = 0  # el nodo 0 es el centro
r_synth, A_synth, V_synth = get_radial_profile(G_synth, center_synth)
references['Radial Synthetic'] = {'G': G_synth, 'center': center_synth, 'r': r_synth, 'A': A_synth, 'V': V_synth}
print(f"      Nodos: {G_synth.number_of_nodes()}")

# ==================== CARGAR GRAFOS DEL MODELO ====================
graph_files = find_graph_files()
selected = get_large_graphs(graph_files)
print(f"\n🎯 Seleccionados {len(selected)} grafos del modelo")

model_profiles = []

for idx, (fname, N) in enumerate(selected):
    print(f"   {idx+1}/{len(selected)}: N={N}...", end='', flush=True)
    try:
        from scipy.sparse import load_npz
        A_mat = load_npz(fname)
        G = nx.from_scipy_sparse_array(A_mat)
        G = get_largest_component(G)
        if G.number_of_nodes() < MIN_NODES:
            print("  -> omitido")
            continue
        
        center, _ = approximate_barycentric_center(G)
        r, A, V = get_radial_profile(G, center)
        
        model_profiles.append({
            'N': N,
            'r': r,
            'A': A,
            'V': V
        })
        print(f"  -> max_r={r[-1]}")
        
    except Exception as e:
        print(f"  -> ERROR: {e}")
        continue

print(f"\n✅ Procesados {len(model_profiles)} grafos del modelo")

# ==================== CÁLCULO DE DISTANCIAS ====================
def normalize_profile(r, A, target_r):
    """Normaliza el perfil A(r) a un rango común."""
    # Interpolar a la grilla de target_r
    from scipy.interpolate import interp1d
    if len(r) < 2:
        return np.zeros_like(target_r)
    f = interp1d(r, A, kind='linear', bounds_error=False, fill_value=0)
    return f(target_r)

def profile_distance(A1, A2):
    """Distancia euclidiana entre dos perfiles normalizados."""
    return np.sqrt(np.sum((A1 - A2)**2))

# Crear grilla común de radios
max_radius = max([len(p['r']) for p in model_profiles] + [len(ref['r']) for ref in references.values()])
common_r = np.arange(max_radius)

# Calcular distancias
distances = []
for model in model_profiles:
    A_model_norm = normalize_profile(model['r'], model['A'], common_r)
    row = {'N': model['N']}
    for ref_name, ref in references.items():
        A_ref_norm = normalize_profile(ref['r'], ref['A'], common_r)
        dist = profile_distance(A_model_norm, A_ref_norm)
        row[ref_name] = dist
    distances.append(row)

# Encontrar mejor coincidencia para cada modelo
best_matches = []
for d in distances:
    ref_names = [k for k in d.keys() if k != 'N']
    best_ref = min(ref_names, key=lambda x: d[x])
    best_dist = d[best_ref]
    best_matches.append({
        'N': d['N'],
        'best': best_ref,
        'distance': best_dist,
        'all_distances': {k: d[k] for k in ref_names}
    })

# ==================== GUARDAR DATOS ====================
with open(COMPARISON_FILE, 'w') as f:
    f.write("# N " + " ".join(references.keys()) + "\n")
    for d in distances:
        f.write(f"{d['N']} ")
        for ref in references.keys():
            f.write(f"{d[ref]:.4f} ")
        f.write("\n")

with open(BEST_MATCH_FILE, 'w') as f:
    f.write("# N best_match distance\n")
    for m in best_matches:
        f.write(f"{m['N']} {m['best']} {m['distance']:.4f}\n")

print(f"💾 Datos guardados en {COMPARISON_FILE} y {BEST_MATCH_FILE}")

# ==================== FIGURAS ====================
# Figura 8: Comparación de perfiles
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Perfiles A(r) (tamaño de esfera)
ax1 = axes[0, 0]
for ref_name, ref in references.items():
    ax1.plot(ref['r'], ref['A'], '--', linewidth=1.5, label=ref_name)
for model in model_profiles[:4]:
    ax1.plot(model['r'], model['A'], '-', linewidth=1, alpha=0.7, label=f'Model N={model["N"]}')
ax1.set_xlabel('Radio r')
ax1.set_ylabel('A(r) (tamaño de capa)')
ax1.set_title('A. Perfiles de esfera')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

# Panel B: Perfiles V(r) (volumen acumulado)
ax2 = axes[0, 1]
for ref_name, ref in references.items():
    ax2.plot(ref['r'], ref['V'], '--', linewidth=1.5, label=ref_name)
for model in model_profiles[:4]:
    ax2.plot(model['r'], model['V'], '-', linewidth=1, alpha=0.7, label=f'Model N={model["N"]}')
ax2.set_xlabel('Radio r')
ax2.set_ylabel('V(r) (volumen acumulado)')
ax2.set_title('B. Perfiles de volumen')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

# Panel C: Comparación de mejor coincidencia
ax3 = axes[1, 0]
match_counts = {}
for m in best_matches:
    match_counts[m['best']] = match_counts.get(m['best'], 0) + 1
refs = list(match_counts.keys())
counts = [match_counts[r] for r in refs]
ax3.bar(refs, counts, color='skyblue', edgecolor='black')
ax3.set_xlabel('Geometría de referencia')
ax3.set_ylabel('Número de grafos')
ax3.set_title('C. Mejor coincidencia por grafo')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# Panel D: Matriz de distancias (mapa de calor)
ax4 = axes[1, 1]
# Preparar matriz de distancias
dist_matrix = []
model_labels = []
for d in distances:
    model_labels.append(f"{d['N']}")
    row = [d[ref] for ref in references.keys()]
    dist_matrix.append(row)
dist_matrix = np.array(dist_matrix)
im = ax4.imshow(dist_matrix, aspect='auto', cmap='viridis')
ax4.set_xticks(range(len(references.keys())))
ax4.set_xticklabels(list(references.keys()), rotation=45, ha='right')
ax4.set_yticks(range(len(model_labels)))
ax4.set_yticklabels(model_labels)
ax4.set_xlabel('Geometría de referencia')
ax4.set_ylabel('Grafo modelo (N)')
ax4.set_title('D. Matriz de distancias')
plt.colorbar(im, ax=ax4, label='Distancia')

plt.tight_layout()
plt.savefig('fig_reference_comparison_profiles.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_reference_comparison_profiles.png', dpi=300, bbox_inches='tight')
print("✅ Figura 8 guardada: fig_reference_comparison_profiles.pdf/png")

# Figura 9: Mapa de calor de distancias
plt.figure(figsize=(10, 8))
plt.imshow(dist_matrix, aspect='auto', cmap='viridis')
plt.colorbar(label='Distancia')
plt.xticks(range(len(references.keys())), list(references.keys()), rotation=45, ha='right')
plt.yticks(range(len(model_labels)), model_labels)
plt.xlabel('Geometría de referencia')
plt.ylabel('Grafo modelo (N)')
plt.title('Distancia entre perfiles radiales')
plt.tight_layout()
plt.savefig('fig_geometry_distance_matrix.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_geometry_distance_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Figura 9 guardada: fig_geometry_distance_matrix.pdf/png")

# ==================== RESUMEN ====================
print("\n" + "="*70)
print(" RESUMEN COMPARACIÓN GEOMÉTRICA")
print("="*70)

# Estadísticas de mejores coincidencias
best_distribution = {}
for m in best_matches:
    best_distribution[m['best']] = best_distribution.get(m['best'], 0) + 1

print(f"\n📊 Mejor coincidencia por grafo:")
for ref, count in sorted(best_distribution.items(), key=lambda x: -x[1]):
    print(f"   {ref}: {count}/{len(model_profiles)} ({100*count/len(model_profiles):.1f}%)")

# Media de distancias
avg_distances = {}
for ref in references.keys():
    avg = np.mean([d[ref] for d in distances])
    avg_distances[ref] = avg
print(f"\n📊 Distancia media a cada referencia:")
for ref, avg in sorted(avg_distances.items(), key=lambda x: x[1]):
    print(f"   {ref}: {avg:.4f}")

# Mejor referencia global
best_global = min(avg_distances, key=avg_distances.get)
print(f"\n🏆 Mejor referencia global: {best_global} (distancia media {avg_distances[best_global]:.4f})")

print("\n✅ Script 4 completado.")