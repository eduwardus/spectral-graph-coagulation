#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_density_of_states_collapse.py

Genera la figura final con dos paneles:
- Panel A: Densidad de estados del Laplaciano (bajos autovalores)
- Panel B: Colapso espectral usando d_s ≈ 3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh
from scipy.stats import linregress
from scipy.interpolate import interp1d
import networkx as nx
import glob
import os
import re
import random
import sys
import time
from tqdm import tqdm

# ==================== CONFIGURACIÓN ====================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

OUTPUT_PDF = "fig_density_of_states_collapse.pdf"
OUTPUT_PNG = "fig_density_of_states_collapse.png"

# Directorios
GRAPH_DIRS = [
    "soup_simulation_phase_transition_v20",
    "soup_simulation_phase_transition_v20/snapshots"
]

# Parámetros
MIN_NODES = 200
MAX_GRAPHS = 60  # Número máximo de grafos a procesar
N_BINS_SIZE = 4  # Número de bins de tamaño para agrupar
N_EIGS = 150     # Número de autovalores a calcular
N_BINS_LAMBDA = 50  # Bins para densidad

# Rango de autovalores para ajuste de d_s
LAMBDA_FIT_MIN = 0.01
LAMBDA_FIT_MAX = 1.0

# ==================== FUNCIONES AUXILIARES ====================
def find_graph_files():
    graph_files = []
    for directory in GRAPH_DIRS:
        if os.path.exists(directory):
            pattern = os.path.join(directory, "*.npz")
            files = glob.glob(pattern)
            print(f"   {directory}: {len(files)} archivos")
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

def get_size_bins(graphs_with_sizes, n_bins=N_BINS_SIZE):
    """Divide los grafos en bins de tamaño para promediar."""
    graphs_with_sizes.sort(key=lambda x: x[1])
    Ns = [s[1] for s in graphs_with_sizes]
    bin_edges = np.percentile(Ns, np.linspace(0, 100, n_bins+1))
    bins = []
    for i in range(len(bin_edges)-1):
        bin_graphs = [(gf, N) for gf, N in graphs_with_sizes if bin_edges[i] <= N < bin_edges[i+1]]
        if bin_graphs:
            bins.append({
                'graphs': bin_graphs,  # lista de (gf, N)
                'N_min': bin_edges[i],
                'N_max': bin_edges[i+1],
                'N_mean': np.mean([N for _, N in bin_graphs])
            })
    return bins

def compute_laplacian_spectrum(G, k=N_EIGS):
    """Calcula los k autovalores más pequeños del Laplaciano."""
    try:
        N = G.number_of_nodes()
        L = nx.laplacian_matrix(G).astype(float)
        if N <= k:
            L_dense = L.toarray()
            eigvals = np.linalg.eigvalsh(L_dense)
            return np.sort(eigvals)[1:]  # eliminar el cero
        else:
            eigvals = eigsh(L, k=k, which='SM', return_eigenvectors=False)
            eigvals = np.sort(eigvals)
            eigvals = eigvals[eigvals > 1e-10]  # eliminar ceros numéricos
            return eigvals
    except Exception as e:
        return None

def compute_density(eigvals, bins=N_BINS_LAMBDA, log_bins=True):
    """Calcula la densidad de estados estimada por histograma."""
    if len(eigvals) < 5:
        return None, None, None
    
    if log_bins:
        min_val = max(eigvals.min(), 1e-6)
        bins_edges = np.logspace(np.log10(min_val), np.log10(eigvals.max()), bins)
    else:
        bins_edges = np.linspace(eigvals.min(), eigvals.max(), bins)
    
    hist, edges = np.histogram(eigvals, bins=bins_edges)
    centers = (edges[:-1] + edges[1:]) / 2
    density = hist / (len(eigvals) * np.diff(edges))
    return centers, density, edges

def estimate_ds_from_density(eigvals, lambda_min=LAMBDA_FIT_MIN, lambda_max=LAMBDA_FIT_MAX):
    """Estima d_s a partir de la densidad de estados en bajos autovalores."""
    centers, density, _ = compute_density(eigvals, bins=30)
    if centers is None:
        return None, 0
    
    mask = (centers >= lambda_min) & (centers <= lambda_max) & (density > 0)
    if np.sum(mask) < 5:
        return None, 0
    
    logx = np.log(centers[mask])
    logy = np.log(density[mask])
    slope, _, r, _, _ = linregress(logx, logy)
    ds = 2 * (slope + 1)
    return ds, r**2

# ==================== PROCESAR GRAFOS ====================
print("\n" + "="*60)
print(" FIGURA: DENSIDAD DE ESTADOS Y COLAPSO ESPECTRAL")
print("="*60)

# Buscar grafos
print("\n🔍 Buscando archivos de grafos...")
graph_files = find_graph_files()
print(f"📁 Encontrados {len(graph_files)} archivos .npz")

# Extraer tamaños y seleccionar muestra
graphs_with_sizes = []
for gf in graph_files:
    N = extract_size_from_filename(gf)
    if N is None:
        try:
            A = load_npz(gf)
            N = A.shape[0]
        except:
            continue
    if N >= MIN_NODES:
        graphs_with_sizes.append((gf, N))

# Selección representativa (muestreo estratificado)
graphs_with_sizes.sort(key=lambda x: x[1])
if len(graphs_with_sizes) > MAX_GRAPHS:
    strata = np.linspace(0, len(graphs_with_sizes)-1, MAX_GRAPHS, dtype=int)
    selected = [graphs_with_sizes[i] for i in strata]
else:
    selected = graphs_with_sizes
print(f"🎯 Seleccionados {len(selected)} grafos (rango: {selected[0][1]} - {selected[-1][1]})")

# Agrupar en bins de tamaño
size_bins = get_size_bins(selected)
print(f"📊 {len(size_bins)} bins de tamaño")

# Procesar cada bin
all_spectra = []  # lista de (N_mean, eigvals)

for bin_data in size_bins:
    print(f"\n   Bin: N ∈ [{bin_data['N_min']:.0f}, {bin_data['N_max']:.0f}]")
    bin_spectra = []
    
    for gf, N in bin_data['graphs']:
        try:
            A = load_npz(gf)
            G = nx.from_scipy_sparse_array(A)
            if not nx.is_connected(G):
                largest = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest)
            
            eigvals = compute_laplacian_spectrum(G)
            if eigvals is not None and len(eigvals) > 10:
                bin_spectra.append(eigvals)
        except Exception as e:
            continue
    
    if bin_spectra:
        # Promediar densidades
        all_centers = []
        all_densities = []
        for spec in bin_spectra:
            centers, density, _ = compute_density(spec, bins=40)
            if centers is not None:
                all_centers.append(centers)
                all_densities.append(density)
        
        # Interpolar a una grilla común para promediar
        if all_centers:
            min_lambda = min([c[0] for c in all_centers])
            max_lambda = max([c[-1] for c in all_centers])
            common_grid = np.logspace(np.log10(max(min_lambda, 1e-4)), np.log10(max_lambda), 50)
            
            interp_densities = []
            for centers, density in zip(all_centers, all_densities):
                f = interp1d(centers, density, kind='linear', bounds_error=False, fill_value=0)
                interp_densities.append(f(common_grid))
            
            mean_density = np.mean(interp_densities, axis=0)
            std_density = np.std(interp_densities, axis=0)
            
            all_spectra.append({
                'N_mean': bin_data['N_mean'],
                'lambda': common_grid,
                'density': mean_density,
                'std': std_density,
                'n_graphs': len(bin_spectra)
            })
            print(f"      {len(bin_spectra)} grafos → densidad promediada")

# ==================== ESTIMACIÓN GLOBAL DE d_s ====================
print("\n" + "="*60)
print(" ESTIMACIÓN GLOBAL DE d_s")
print("="*60)

ds_global = 2.97  # valor conocido de análisis previos
ds_errors = []

# Alternativamente, estimar a partir de los datos actuales
for spec in all_spectra:
    # Reconstruir puntos de densidad a partir de la curva promediada
    mask = (spec['lambda'] >= LAMBDA_FIT_MIN) & (spec['lambda'] <= LAMBDA_FIT_MAX) & (spec['density'] > 0)
    if np.sum(mask) > 5:
        logx = np.log(spec['lambda'][mask])
        logy = np.log(spec['density'][mask])
        slope, _, r, _, _ = linregress(logx, logy)
        ds = 2 * (slope + 1)
        ds_errors.append(ds)
        print(f"   N ≈ {spec['N_mean']:.0f}: d_s = {ds:.3f} (R² = {r**2:.3f})")

if ds_errors:
    ds_global = np.mean(ds_errors)
    print(f"\n📊 d_s global estimado: {ds_global:.3f} ± {np.std(ds_errors):.3f}")

# ==================== FIGURA: DOS PANELES ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Colores para los bins
colors = plt.cm.viridis(np.linspace(0, 1, len(all_spectra)))

# ==================== PANEL A: DENSIDAD DE ESTADOS ====================
for i, spec in enumerate(all_spectra):
    ax1.loglog(spec['lambda'], spec['density'],
               color=colors[i], linewidth=1.5,
               label=f'N ≈ {spec["N_mean"]:.0f} (n={spec["n_graphs"]})')

# Línea de referencia teórica con d_s global
lambda_theory = np.logspace(-2, 0, 100)
rho_theory = lambda_theory ** (ds_global/2 - 1)
rho_theory = rho_theory / rho_theory.max() * 1e-2  # escalado para visualización
ax1.loglog(lambda_theory, rho_theory, 'k--', linewidth=1.5,
           label=r'$\rho \sim \lambda^{d_s/2-1}$, $d_s\approx$' + f'{ds_global:.2f}')

ax1.set_xlabel(r'Laplacian eigenvalue $\lambda$')
ax1.set_ylabel(r'Density of states $\rho(\lambda)$')
ax1.set_xlim(1e-3, 1e1)
ax1.grid(True, alpha=0.2)
ax1.legend(loc='lower left', fontsize=8)

# ==================== PANEL B: COLAPSO ESPECTRAL ====================
for i, spec in enumerate(all_spectra):
    # Reescalado horizontal: λ → λ * N^{2/d_s}
    scaled_lambda = spec['lambda'] * (spec['N_mean'] ** (2.0 / ds_global))
    ax2.loglog(scaled_lambda, spec['density'],
               color=colors[i], linewidth=1.5, alpha=0.7)

ax2.set_xlabel(r'Rescaled eigenvalue $\lambda N^{2/d_s}$')
ax2.set_ylabel(r'Density of states $\rho(\lambda)$')
ax2.set_xlim(1e-3, 1e1)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"\n✅ Figura guardada: {OUTPUT_PDF}")

# ==================== RESUMEN NUMÉRICO ====================
print("\n" + "="*60)
print(" RESUMEN NUMÉRICO")
print("="*60)
print(f"Número de grafos usados: {len(selected)}")
print(f"Rango de tamaños: {selected[0][1]} - {selected[-1][1]}")
print(f"d_s global ajustado: {ds_global:.3f} ± {np.std(ds_errors):.3f}" if ds_errors else f"d_s global: {ds_global:.3f}")
print(f"Rango de autovalores para ajuste: [{LAMBDA_FIT_MIN}, {LAMBDA_FIT_MAX}]")

# Calcular calidad del colapso (varianza entre curvas en una grilla común)
if len(all_spectra) > 1:
    # Interpolar todas las curvas reescaladas a una grilla común
    min_lambda_scaled = min([s['lambda'][0] * (s['N_mean'] ** (2.0/ds_global)) for s in all_spectra])
    max_lambda_scaled = max([s['lambda'][-1] * (s['N_mean'] ** (2.0/ds_global)) for s in all_spectra])
    common_grid = np.logspace(np.log10(max(min_lambda_scaled, 1e-4)), np.log10(max_lambda_scaled), 100)
    
    interp_densities = []
    for spec in all_spectra:
        scaled_lambda = spec['lambda'] * (spec['N_mean'] ** (2.0/ds_global))
        f = interp1d(scaled_lambda, spec['density'], kind='linear', bounds_error=False, fill_value=0)
        interp_densities.append(f(common_grid))
    
    interp_array = np.array(interp_densities)
    collapse_variance = np.var(interp_array, axis=0).mean()
    print(f"Calidad del colapso (varianza media): {collapse_variance:.6f}")

# ==================== GUARDAR DATOS RESUMEN ====================
summary_data = []
for spec in all_spectra:
    summary_data.append([spec['N_mean'], spec['n_graphs']])
if summary_data:
    summary_data = np.array(summary_data)
    np.savetxt('spectral_density_summary.dat', summary_data,
               header='N_mean n_graphs', fmt='%.0f %d')
    print("\n💾 Datos resumen guardados en spectral_density_summary.dat")