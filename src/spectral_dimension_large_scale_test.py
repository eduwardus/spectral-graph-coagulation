# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 00:40:58 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectral_dimension_large_scale_test.py

Análisis de la dimensión espectral en grafos grandes (N > 5000)
para verificar la convergencia hacia un valor asintótico.
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
plt.rcParams['font.size'] = 12

OUTPUT_PDF = "fig_spectral_dimension_large_scale.pdf"
OUTPUT_PNG = "fig_spectral_dimension_large_scale.png"

# Archivos de datos
DATA_FILE = "spectral_dimension_vs_size.dat"
ALT_DATA_FILE = "spectral_density_data.txt"

# Parámetros de filtrado
MIN_NODES = 5000  # SÓLO GRAFOS GRANDES
MIN_QUALITY = 0.1
DS_MIN = 0.5
DS_MAX = 5.0

# ==================== CARGAR DATOS ====================
def load_data():
    """Carga los datos de dimensión espectral."""
    
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
            print("   Error al cargar, intentando archivo alternativo...")
    
    # Intentar archivo procesado
    processed_file = "spectral_dimension_vs_size_processed.dat"
    if os.path.exists(processed_file):
        print(f"📊 Cargando datos desde {processed_file}")
        try:
            data = np.loadtxt(processed_file, comments='#')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            N = data[:, 0]
            ds = data[:, 1]
            quality = data[:, 2] if data.shape[1] > 2 else np.ones_like(N)
            return N, ds, quality
        except:
            pass
    
    print("❌ No se encontraron datos")
    return None, None, None

# ==================== MODELOS ASINTÓTICOS ====================
def constant_model(N, d_inf):
    """Modelo constante: d_s = d_inf"""
    return d_inf * np.ones_like(N)

def power_model(N, d_inf, a, b):
    """Modelo de decaimiento: d_s = d_inf - a * N^{-b}"""
    return d_inf - a * N**(-b)

def log_model(N, d_inf, a):
    """Modelo logarítmico: d_s = d_inf - a / log(N)"""
    return d_inf - a / np.log(N)

# ==================== ANÁLISIS ====================
def main():
    print("\n" + "="*70)
    print(" ANÁLISIS DE DIMENSIÓN ESPECTRAL EN GRAFOS GRANDES (N > 5000)")
    print("="*70)
    
    # Cargar datos
    N, ds, quality = load_data()
    if N is None:
        print("❌ No se pudieron cargar los datos")
        sys.exit(1)
    
    # Filtrar
    mask = (N >= MIN_NODES) & (quality > MIN_QUALITY) & (ds >= DS_MIN) & (ds <= DS_MAX)
    N_large = N[mask]
    ds_large = ds[mask]
    
    print(f"\n📊 Datos después de filtrar (N ≥ {MIN_NODES}):")
    print(f"   Grafos grandes: {len(N_large)}")
    print(f"   Rango de tamaños: {min(N_large):.0f} - {max(N_large):.0f}")
    
    if len(N_large) < 5:
        print("\n⚠️  Pocos grafos grandes. Reduciendo umbral...")
        mask = (N >= 3000) & (quality > MIN_QUALITY) & (ds >= DS_MIN) & (ds <= DS_MAX)
        N_large = N[mask]
        ds_large = ds[mask]
        print(f"   Nuevo umbral (N ≥ 3000): {len(N_large)} grafos")
        
        if len(N_large) < 5:
            print("❌ Insuficientes datos para análisis")
            sys.exit(1)
    
    # ==================== ESTADÍSTICAS BÁSICAS ====================
    mean_ds = np.mean(ds_large)
    std_ds = np.std(ds_large)
    median_ds = np.median(ds_large)
    
    print(f"\n" + "="*70)
    print(" ESTADÍSTICAS DE GRAFOS GRANDES")
    print("="*70)
    print(f"\n📊 Dimensión espectral en grafos grandes:")
    print(f"   Media: {mean_ds:.3f} ± {std_ds:.3f}")
    print(f"   Mediana: {median_ds:.3f}")
    print(f"   Mínimo: {np.min(ds_large):.3f}")
    print(f"   Máximo: {np.max(ds_large):.3f}")
    
    # ==================== DEPENDENCIA RESIDUAL CON N ====================
    print(f"\n" + "="*70)
    print(" DEPENDENCIA RESIDUAL CON EL TAMAÑO")
    print("="*70)
    
    # Ajuste lineal en log-log para ver tendencia residual
    logN = np.log(N_large)
    log_ds = np.log(ds_large)
    slope, intercept, r, p, stderr = linregress(logN, log_ds)
    r2 = r**2
    
    print(f"\n📈 Ajuste lineal en log-log: log(d_s) = {slope:.4f}·log(N) + {intercept:.4f}")
    print(f"   R² = {r2:.4f}")
    print(f"   p-value = {p:.4e}")
    
    if abs(slope) < 0.05:
        print("   ✅ Pendiente cercana a cero → d_s ≈ constante en este rango")
    elif slope > 0:
        print(f"   ⚠️ Pendiente positiva → d_s aún crece (pendiente {slope:.3f})")
    else:
        print(f"   ⚠️ Pendiente negativa → d_s decrece (pendiente {slope:.3f})")
    
    # ==================== AJUSTES ASINTÓTICOS ====================
    print(f"\n" + "="*70)
    print(" AJUSTES ASINTÓTICOS")
    print("="*70)
    
    # Ordenar para ajustes
    order = np.argsort(N_large)
    N_sorted = N_large[order]
    ds_sorted = ds_large[order]
    
    # Modelo constante
    mean_const = np.mean(ds_sorted)
    std_const = np.std(ds_sorted)
    print(f"\n📈 Modelo constante: d_s = {mean_const:.3f} ± {std_const:.3f}")
    
    # Modelo de decaimiento potencial (d_s = d_inf - a·N^{-b})
    try:
        popt_pow, pcov_pow = curve_fit(power_model, N_sorted, ds_sorted,
                                       p0=[mean_const, 1.0, 0.5], maxfev=5000)
        ds_fit_pow = power_model(N_sorted, *popt_pow)
        ss_res = np.sum((ds_sorted - ds_fit_pow)**2)
        ss_tot = np.sum((ds_sorted - mean_const)**2)
        r2_pow = 1 - ss_res/ss_tot
        print(f"\n📈 Modelo potencial: d_s = {popt_pow[0]:.3f} - {popt_pow[1]:.3f}·N^{{-{popt_pow[2]:.3f}}}")
        print(f"   R² = {r2_pow:.4f}")
        if popt_pow[0] > 0:
            print(f"   Asíntota estimada: d_inf = {popt_pow[0]:.3f} ± {np.sqrt(pcov_pow[0,0]):.3f}")
    except Exception as e:
        print(f"\n⚠️ No se pudo ajustar modelo potencial: {e}")
        r2_pow = -1
    
    # Modelo logarítmico
    try:
        popt_log, pcov_log = curve_fit(log_model, N_sorted, ds_sorted,
                                       p0=[mean_const, 1.0], maxfev=5000)
        ds_fit_log = log_model(N_sorted, *popt_log)
        ss_res = np.sum((ds_sorted - ds_fit_log)**2)
        r2_log = 1 - ss_res/ss_tot
        print(f"\n📈 Modelo logarítmico: d_s = {popt_log[0]:.3f} - {popt_log[1]:.3f}/log(N)")
        print(f"   R² = {r2_log:.4f}")
        print(f"   Asíntota estimada: d_inf = {popt_log[0]:.3f} ± {np.sqrt(pcov_log[0,0]):.3f}")
    except Exception as e:
        print(f"\n⚠️ No se pudo ajustar modelo logarítmico: {e}")
        r2_log = -1
    
    # ==================== MEJOR ESTIMACIÓN ====================
    print(f"\n" + "="*70)
    print(" CONCLUSIÓN")
    print("="*70)
    
    # Usar la media de los últimos puntos como mejor estimación
    n_last = min(10, len(ds_sorted))
    ds_last = ds_sorted[-n_last:]
    N_last = N_sorted[-n_last:]
    mean_last = np.mean(ds_last)
    std_last = np.std(ds_last)
    
    print(f"\n📊 Últimos {n_last} grafos (N > {N_last[0]:.0f}):")
    print(f"   d_s = {mean_last:.3f} ± {std_last:.3f}")
    
    # Determinar la mejor estimación
    if r2_pow > 0.5 and popt_pow[0] > 0:
        best_estimate = popt_pow[0]
        best_error = np.sqrt(pcov_pow[0,0])
        best_method = "potencial"
    elif r2_log > 0.5 and popt_log[0] > 0:
        best_estimate = popt_log[0]
        best_error = np.sqrt(pcov_log[0,0])
        best_method = "logarítmico"
    else:
        best_estimate = mean_last
        best_error = std_last / np.sqrt(n_last)
        best_method = "últimos puntos"
    
    print(f"\n🏆 Mejor estimación asintótica ({best_method}):")
    print(f"   d_s = {best_estimate:.3f} ± {best_error:.3f}")
    
    # ==================== FIGURA ====================
    plt.figure(figsize=(10, 8))
    
    # Puntos individuales
    plt.scatter(N_large, ds_large, alpha=0.5, s=15, c='gray', label=f'N ≥ {MIN_NODES}')
    
    # Media y banda de 1σ
    plt.axhline(y=mean_last, color='red', linestyle='-', linewidth=2,
                label=f'Media últimos {n_last} puntos: {mean_last:.2f}')
    plt.fill_between([min(N_large), max(N_large)], 
                     [mean_last - std_last, mean_last - std_last],
                     [mean_last + std_last, mean_last + std_last],
                     color='red', alpha=0.2, label='±1σ')
    
    # Línea asintótica
    plt.axhline(y=3.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                label='d_s = 3 (referencia)')
    
    plt.xscale('log')
    plt.xlabel('Cluster size $N$', fontsize=14)
    plt.ylabel('Spectral dimension $d_s$', fontsize=14)
    plt.title(f'Dimensión espectral en grafos grandes (N ≥ {MIN_NODES})', fontsize=16)
    plt.ylim(0, 5)
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figura guardada: {OUTPUT_PDF}")
    
    # ==================== VERIFICACIÓN DE CONVERGENCIA ====================
    print(f"\n" + "="*70)
    print(" VERIFICACIÓN DE CONVERGENCIA")
    print("="*70)
    
    # Test de tendencia: dividir en dos mitades y comparar
    n_half = len(ds_sorted) // 2
    if n_half > 0:
        first_half = ds_sorted[:n_half]
        second_half = ds_sorted[-n_half:]
        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)
        diff = abs(mean_second - mean_first)
        
        print(f"\n📊 Comparación primera vs segunda mitad:")
        print(f"   Primera mitad (N < {N_sorted[n_half]:.0f}): d_s = {mean_first:.3f}")
        print(f"   Segunda mitad (N > {N_sorted[-n_half]:.0f}): d_s = {mean_second:.3f}")
        print(f"   Diferencia: {diff:.3f}")
        
        if diff < 0.2:
            print("   ✅ Convergencia estable → d_s se estabiliza")
        else:
            print(f"   ⚠️ Tendencia aún visible (diferencia {diff:.3f})")
    
    # ==================== GUARDAR RESULTADOS ====================
    with open('spectral_dimension_large_scale_results.txt', 'w') as f:
        f.write("# Resultados del análisis en grafos grandes\n")
        f.write(f"# Umbral: N >= {MIN_NODES}\n")
        f.write(f"# Grafos analizados: {len(N_large)}\n")
        f.write(f"# Rango de tamaños: {min(N_large)} - {max(N_large)}\n\n")
        f.write(f"Media total: {mean_ds:.4f} ± {std_ds:.4f}\n")
        f.write(f"Mediana: {median_ds:.4f}\n")
        f.write(f"Últimos {n_last} grafos: {mean_last:.4f} ± {std_last:.4f}\n")
        f.write(f"Mejor estimación ({best_method}): {best_estimate:.4f} ± {best_error:.4f}\n")
    
    print(f"\n💾 Resultados guardados en spectral_dimension_large_scale_results.txt")

if __name__ == "__main__":
    main()