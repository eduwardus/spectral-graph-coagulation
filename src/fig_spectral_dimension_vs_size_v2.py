# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:35:31 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_spectral_dimension_vs_size_v2.py

Genera la figura final de dimensión espectral vs tamaño del cluster.
- Puntos individuales en gris
- Promedios por bins logarítmicos con barras de error
- Ajuste asintótico hacia d_s ≈ 3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import os
import sys
import glob
import re
import random
import time

# ==================== CONFIGURACIÓN ====================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

OUTPUT_PDF = "fig_spectral_dimension_vs_size_v2.pdf"
OUTPUT_PNG = "fig_spectral_dimension_vs_size_v2.png"

# Archivos de datos
DATA_FILE = "spectral_dimension_vs_size.dat"
ALT_DATA_FILE = "spectral_density_data.txt"

# Parámetros de filtrado
MIN_NODES = 100
MIN_QUALITY = 0.1
DS_MIN = 0.5
DS_MAX = 5.0

# Binning
N_BINS = 12

# ==================== MODELOS ASINTÓTICOS ====================
def model_log(N, d_inf, a):
    """d_s(N) = d_inf - a / log(N)"""
    return d_inf - a / np.log(N)

def model_power(N, d_inf, a, b):
    """d_s(N) = d_inf - a * N^{-b}"""
    return d_inf - a * N**(-b)

# ==================== CARGAR O RECONSTRUIR DATOS ====================
def load_or_recompute_data():
    """Carga los datos existentes o los reconstruye si no existen."""
    
    if os.path.exists(DATA_FILE):
        print(f"📊 Cargando datos desde {DATA_FILE}")
        try:
            data = np.loadtxt(DATA_FILE, comments='#')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            # Detectar columnas
            if data.shape[1] >= 3:
                N = data[:, 0]
                ds = data[:, 1]
                quality = data[:, 2] if data.shape[1] > 2 else np.ones_like(N)
            else:
                N = data[:, 0]
                ds = data[:, 1]
                quality = np.ones_like(N)
            return N, ds, quality
        except:
            print("   Error al cargar, reconstruyendo...")
    
    print("📊 Reconstruyendo datos desde archivos .npz...")
    
    # Buscar grafos
    graph_files = []
    for directory in ["soup_simulation_phase_transition_v20",
                      "soup_simulation_phase_transition_v20/snapshots"]:
        if os.path.exists(directory):
            pattern = os.path.join(directory, "*.npz")
            files = glob.glob(pattern)
            graph_files.extend(files)
    
    if not graph_files:
        print("❌ No se encontraron archivos .npz")
        sys.exit(1)
    
    # Función para estimar d_s (simplificada)
    def estimate_ds_from_graph(G, max_walkers=100, max_steps=200):
        """Estimación rápida de d_s usando random walk."""
        try:
            N = G.number_of_nodes()
            if N < MIN_NODES:
                return None, 0
            
            # Random walk para estimar probabilidad de retorno
            nodes = list(G.nodes())
            if len(nodes) < max_walkers:
                origins = nodes
            else:
                origins = random.sample(nodes, max_walkers)
            
            P_t = np.zeros(max_steps+1)
            for origin in origins:
                pos = origin
                P_t[0] += 1
                for t in range(1, max_steps+1):
                    nb = list(G.neighbors(pos))
                    if nb:
                        pos = random.choice(nb)
                    if pos == origin:
                        P_t[t] += 1
            P_t = P_t / len(origins)
            
            # Ajuste en régimen difusivo
            t = np.arange(1, max_steps)
            mask = (t > 5) & (P_t[1:] > 0)
            if np.sum(mask) < 5:
                return None, 0
            
            logt = np.log(t[mask])
            logP = np.log(P_t[1:][mask])
            slope, _, r, _, _ = linregress(logt, logP)
            ds = -2 * slope
            quality = r**2
            return ds, quality
        except:
            return None, 0
    
    # Procesar grafos
    N_vals = []
    ds_vals = []
    qual_vals = []
    
    for gf in graph_files:
        try:
            from scipy.sparse import load_npz
            import networkx as nx
            
            # Extraer tamaño
            match = re.search(r'_N(\d+)_', os.path.basename(gf))
            if match:
                N = int(match.group(1))
            else:
                A = load_npz(gf)
                N = A.shape[0]
            
            if N < MIN_NODES:
                continue
            
            A = load_npz(gf)
            G = nx.from_scipy_sparse_array(A)
            if not nx.is_connected(G):
                largest = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest)
            
            ds, qual = estimate_ds_from_graph(G)
            if ds is not None and qual > MIN_QUALITY and DS_MIN <= ds <= DS_MAX:
                N_vals.append(N)
                ds_vals.append(ds)
                qual_vals.append(qual)
                
        except Exception as e:
            continue
    
    print(f"   Procesados {len(N_vals)} grafos válidos")
    return np.array(N_vals), np.array(ds_vals), np.array(qual_vals)

# ==================== PROCESAMIENTO ====================
print("\n" + "="*60)
print(" FIGURA: DIMENSIÓN ESPECTRAL VS TAMAÑO")
print("="*60)

N, ds, quality = load_or_recompute_data()

# Filtrar
mask = (N >= MIN_NODES) & (quality > MIN_QUALITY) & (ds >= DS_MIN) & (ds <= DS_MAX)
N = N[mask]
ds = ds[mask]

print(f"\n📊 Datos después de filtrar:")
print(f"   Grafos válidos: {len(N)}")
print(f"   Rango de tamaños: {min(N):.0f} - {max(N):.0f}")
print(f"   ds: media = {np.mean(ds):.3f}, mediana = {np.median(ds):.3f}")

# Ordenar
order = np.argsort(N)
N = N[order]
ds = ds[order]

# ==================== BINNING LOGARÍTMICO ====================
bins = np.logspace(np.log10(max(min(N), 10)), np.log10(max(N)), N_BINS)
bin_centers = []
bin_means = []
bin_stds = []
bin_counts = []

for i in range(len(bins)-1):
    mask_bin = (N >= bins[i]) & (N < bins[i+1])
    if np.sum(mask_bin) >= 3:
        bin_centers.append(np.exp(np.mean(np.log(N[mask_bin]))))
        bin_means.append(np.mean(ds[mask_bin]))
        bin_stds.append(np.std(ds[mask_bin]) / np.sqrt(np.sum(mask_bin)))  # error estándar
        bin_counts.append(np.sum(mask_bin))

bin_centers = np.array(bin_centers)
bin_means = np.array(bin_means)
bin_stds = np.array(bin_stds)

print(f"\n📊 Binning: {len(bin_centers)} bins")

# ==================== AJUSTE ASINTÓTICO ====================
best_model = None
best_r2 = -np.inf
best_params = None

# Modelo logarítmico
try:
    popt_log, pcov_log = curve_fit(model_log, bin_centers, bin_means,
                                   p0=[3.0, 2.0], maxfev=5000)
    y_pred_log = model_log(bin_centers, *popt_log)
    ss_res = np.sum((bin_means - y_pred_log)**2)
    ss_tot = np.sum((bin_means - np.mean(bin_means))**2)
    r2_log = 1 - ss_res/ss_tot
    print(f"\n📈 Modelo logarítmico: d_s = {popt_log[0]:.3f} - {popt_log[1]:.3f}/log(N)")
    print(f"   R² = {r2_log:.4f}")
    if r2_log > best_r2:
        best_r2 = r2_log
        best_model = "log"
        best_params = popt_log
except:
    r2_log = -np.inf

# Modelo potencial
try:
    popt_pow, pcov_pow = curve_fit(model_power, bin_centers, bin_means,
                                   p0=[3.0, 2.0, 0.5], maxfev=5000)
    y_pred_pow = model_power(bin_centers, *popt_pow)
    ss_res = np.sum((bin_means - y_pred_pow)**2)
    r2_pow = 1 - ss_res/ss_tot
    print(f"\n📈 Modelo potencial: d_s = {popt_pow[0]:.3f} - {popt_pow[1]:.3f}·N^{{-{popt_pow[2]:.3f}}}")
    print(f"   R² = {r2_pow:.4f}")
    if r2_pow > best_r2:
        best_r2 = r2_pow
        best_model = "power"
        best_params = popt_pow
except:
    r2_pow = -np.inf

print(f"\n🏆 Mejor modelo: {best_model.upper()} (R² = {best_r2:.4f})")

# ==================== FIGURA ====================
plt.figure(figsize=(8, 6))

# Puntos individuales
plt.scatter(N, ds, alpha=0.2, s=8, c='gray', label='Individual graphs')

# Promedios por bin con error estándar
plt.errorbar(bin_centers, bin_means, yerr=bin_stds,
             fmt='o', capsize=3, markersize=6, color='red',
             label=f'Bin means (n={bin_counts[0]}–{bin_counts[-1]})')

# Línea de ajuste
if best_model == "log":
    N_fit = np.logspace(np.log10(min(N)), np.log10(max(N)), 100)
    ds_fit = model_log(N_fit, *best_params)
    plt.plot(N_fit, ds_fit, 'k--', linewidth=1.5,
             label=f'Fit: $d_s = {best_params[0]:.2f} - {best_params[1]:.2f}/\\log N$')
elif best_model == "power":
    N_fit = np.logspace(np.log10(min(N)), np.log10(max(N)), 100)
    ds_fit = model_power(N_fit, *best_params)
    plt.plot(N_fit, ds_fit, 'k--', linewidth=1.5,
             label=f'Fit: $d_s = {best_params[0]:.2f} - {best_params[1]:.2f} N^{{-{best_params[2]:.2f}}}$')

# Línea asintótica d_s = 3
plt.axhline(y=3.0, color='gray', linestyle='--', linewidth=1.5,
            label='$d_s = 3$')

plt.xscale('log')
plt.xlabel(r'Cluster size $N$')
plt.ylabel(r'Spectral dimension $d_s$')
plt.ylim(0.5, 4.5)
plt.grid(True, alpha=0.2)
plt.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"\n✅ Figura guardada: {OUTPUT_PDF}")

# ==================== RESUMEN NUMÉRICO ====================
print("\n" + "="*60)
print(" RESUMEN NUMÉRICO")
print("="*60)
print(f"Total de grafos válidos: {len(N)}")
print(f"Rango de tamaños: {min(N):.0f} - {max(N):.0f}")
print(f"ds media: {np.mean(ds):.3f} ± {np.std(ds):.3f}")
print(f"ds mediana: {np.median(ds):.3f}")

# Último tercio de tamaños
third = len(N) // 3
large_ds = ds[-third:]
print(f"ds en último tercio (N > {N[-third]:.0f}): {np.mean(large_ds):.3f} ± {np.std(large_ds):.3f}")

if best_model == "log":
    print(f"Ajuste asintótico: d_inf = {best_params[0]:.3f} ± {np.sqrt(pcov_log[0,0]):.3f}")
elif best_model == "power":
    print(f"Ajuste asintótico: d_inf = {best_params[0]:.3f} ± {np.sqrt(pcov_pow[0,0]):.3f}")
print(f"R² del ajuste: {best_r2:.4f}")

# Guardar datos procesados para reutilización
np.savetxt('spectral_dimension_vs_size_processed.dat',
           np.column_stack((N, ds, quality[mask])),
           header='N ds quality', fmt='%d %.6f %.4f')
print("\n💾 Datos procesados guardados en spectral_dimension_vs_size_processed.dat")