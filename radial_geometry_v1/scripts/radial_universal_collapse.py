# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 00:34:14 2026

@author: eggra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radial_universal_collapse.py

Script 6: Colapso radial universal.
- Carga perfiles radiales de grafos grandes.
- Reescala por radio máximo y volumen normalizado.
- Busca una curva universal empírica.
- Genera figuras de colapso.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13

# Archivos de entrada (generados por scripts anteriores)
PROFILE_FILES = glob.glob("geometry_N*.dat")
CURVATURE_FILES = glob.glob("curvature_N*.dat")
ANISOTROPY_FILE = "radial_anisotropy_data.dat"

# Archivos de salida
COLLAPSE_DATA_FILE = "radial_collapse_data.dat"
FIT_PARAMS_FILE = "radial_collapse_fit_params.txt"

# ==================== FUNCIONES ====================
def load_profile(filename):
    """Carga perfil radial de geometry_N*.dat"""
    data = np.loadtxt(filename)
    # Columnas: r A(r) V(r) deg_mean deg_std R_eff cycle_dens
    r = data[:, 0].astype(int)
    A = data[:, 1]
    V = data[:, 2]
    return r, A, V

def load_curvature(filename):
    """Carga curvatura radial de curvature_N*.dat"""
    data = np.loadtxt(filename)
    # Columnas: r curvature count
    r = data[:, 0].astype(int)
    curvature = data[:, 1]
    return r, curvature

def extract_N_from_filename(filename):
    """Extrae N del nombre del archivo"""
    match = re.search(r'N(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def universal_function(x, a, b, c):
    """Función empírica para el colapso: y = a * x^b * (1-x)^c"""
    return a * (x ** b) * ((1 - x) ** c)

def collapse_profiles(profiles):
    """
    Colapsa los perfiles A(r) y V(r) reescalando por radio máximo y volumen.
    Retorna arrays colapsados y estadísticas.
    """
    collapsed_A = []
    collapsed_V = []
    collapsed_r_norm = []
    N_vals = []
    
    for r, A, V, N in profiles:
        r_max = r[-1]
        if r_max == 0:
            continue
        
        r_norm = r / r_max
        A_norm = A / np.max(A)
        V_norm = V / N
        
        collapsed_r_norm.extend(r_norm)
        collapsed_A.extend(A_norm)
        collapsed_V.extend(V_norm)
        N_vals.append(N)
    
    collapsed_r_norm = np.array(collapsed_r_norm)
    collapsed_A = np.array(collapsed_A)
    collapsed_V = np.array(collapsed_V)
    
    # Crear grilla común para el colapso
    common_grid = np.linspace(0, 1, 100)
    
    # Interpolar cada perfil para obtener el colapso promedio
    all_A_interp = []
    all_V_interp = []
    
    for r, A, V, N in profiles:
        r_max = r[-1]
        if r_max == 0:
            continue
        r_norm = r / r_max
        A_norm = A / np.max(A)
        V_norm = V / N
        
        # Interpolación
        f_A = interp1d(r_norm, A_norm, kind='linear', bounds_error=False, fill_value=0)
        f_V = interp1d(r_norm, V_norm, kind='linear', bounds_error=False, fill_value=1)
        
        all_A_interp.append(f_A(common_grid))
        all_V_interp.append(f_V(common_grid))
    
    mean_A = np.mean(all_A_interp, axis=0)
    std_A = np.std(all_A_interp, axis=0)
    mean_V = np.mean(all_V_interp, axis=0)
    std_V = np.std(all_V_interp, axis=0)
    
    return common_grid, mean_A, std_A, mean_V, std_V, collapsed_r_norm, collapsed_A, collapsed_V

def fit_universal_curve(x, y):
    """Ajusta la función universal a los datos colapsados."""
    # Filtrar puntos válidos (evitar x=0 y x=1)
    mask = (x > 0.02) & (x < 0.98) & (y > 0.01)
    x_fit = x[mask]
    y_fit = y[mask]
    
    if len(x_fit) < 10:
        return None, None
    
    try:
        popt, pcov = curve_fit(universal_function, x_fit, y_fit, 
                               p0=[1.0, 0.5, 0.5], maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except:
        return None, None

# ==================== PROCESAMIENTO ====================
print("="*70)
print(" BLOQUE 6: COLAPSO RADIAL UNIVERSAL")
print("="*70)

# Cargar perfiles radiales
profiles = []
for f in PROFILE_FILES:
    N = extract_N_from_filename(f)
    if N is None:
        continue
    if N >= 5000:  # Solo grafos grandes
        r, A, V = load_profile(f)
        profiles.append((r, A, V, N))

print(f"📊 Cargados {len(profiles)} perfiles radiales")

if len(profiles) == 0:
    print("❌ No se encontraron perfiles radiales")
    exit(1)

# Colapso de perfiles
common_grid, mean_A, std_A, mean_V, std_V, collapsed_r, collapsed_A, collapsed_V = collapse_profiles(profiles)

# Ajuste universal para A(r)
popt_A, perr_A = fit_universal_curve(common_grid, mean_A)
# Ajuste universal para V(r)
popt_V, perr_V = fit_universal_curve(common_grid, mean_V)

# ==================== GUARDAR DATOS ====================
# Datos colapsados
with open(COLLAPSE_DATA_FILE, 'w') as f:
    f.write("# r_norm mean_A std_A mean_V std_V\n")
    for i in range(len(common_grid)):
        f.write(f"{common_grid[i]:.6f} {mean_A[i]:.6f} {std_A[i]:.6f} {mean_V[i]:.6f} {std_V[i]:.6f}\n")

# Parámetros del ajuste
with open(FIT_PARAMS_FILE, 'w') as f:
    f.write("# Parámetros del ajuste universal: A(r) = a * x^b * (1-x)^c\n")
    if popt_A is not None:
        f.write(f"A(r): a={popt_A[0]:.4f} ± {perr_A[0]:.4f}, "
                f"b={popt_A[1]:.4f} ± {perr_A[1]:.4f}, "
                f"c={popt_A[2]:.4f} ± {perr_A[2]:.4f}\n")
    else:
        f.write("A(r): ajuste fallido\n")
    
    f.write("\n# Parámetros para V(r)\n")
    if popt_V is not None:
        f.write(f"V(r): a={popt_V[0]:.4f} ± {perr_V[0]:.4f}, "
                f"b={popt_V[1]:.4f} ± {perr_V[1]:.4f}, "
                f"c={popt_V[2]:.4f} ± {perr_V[2]:.4f}\n")
    else:
        f.write("V(r): ajuste fallido\n")

print(f"💾 Datos guardados en {COLLAPSE_DATA_FILE} y {FIT_PARAMS_FILE}")

# ==================== FIGURAS ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Figura A: Colapso de A(r)
ax = axes[0, 0]
ax.errorbar(common_grid, mean_A, yerr=std_A, fmt='o-', markersize=2, 
            capsize=2, color='blue', alpha=0.7, label='Perfil colapsado')
if popt_A is not None:
    y_fit = universal_function(common_grid, *popt_A)
    ax.plot(common_grid, y_fit, 'r--', linewidth=2, 
            label=f'Ajuste: $a x^{popt_A[1]:.2f} (1-x)^{popt_A[2]:.2f}$')
ax.set_xlabel('Radio normalizado $r / r_{max}$')
ax.set_ylabel('Tamaño de capa normalizado $A(r)/A_{max}$')
ax.set_title('A. Colapso del perfil de capas')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Figura B: Colapso de V(r)
ax = axes[0, 1]
ax.errorbar(common_grid, mean_V, yerr=std_V, fmt='s-', markersize=2,
            capsize=2, color='green', alpha=0.7, label='Perfil colapsado')
if popt_V is not None:
    y_fit = universal_function(common_grid, *popt_V)
    ax.plot(common_grid, y_fit, 'r--', linewidth=2,
            label=f'Ajuste: $a x^{popt_V[1]:.2f} (1-x)^{popt_V[2]:.2f}$')
ax.set_xlabel('Radio normalizado $r / r_{max}$')
ax.set_ylabel('Volumen normalizado $V(r)/N$')
ax.set_title('B. Colapso del perfil de volumen')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Figura C: Dispersión entre grafos
ax = axes[1, 0]
for i, (r, A, V, N) in enumerate(profiles[:8]):  # Mostrar hasta 8
    r_max = r[-1]
    if r_max == 0:
        continue
    r_norm = r / r_max
    A_norm = A / np.max(A)
    ax.plot(r_norm, A_norm, '-', alpha=0.5, linewidth=0.8, label=f'N={N}')
ax.set_xlabel('Radio normalizado $r / r_{max}$')
ax.set_ylabel('$A(r)/A_{max}$')
ax.set_title('C. Perfiles individuales')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, ncol=2)

# Figura D: Varianza del colapso
ax = axes[1, 1]
ax.fill_between(common_grid, mean_A - std_A, mean_A + std_A, alpha=0.3, color='blue')
ax.fill_between(common_grid, mean_V - std_V, mean_V + std_V, alpha=0.3, color='green')
ax.plot(common_grid, std_A, 'b-', label='Desviación A(r)')
ax.plot(common_grid, std_V, 'g-', label='Desviación V(r)')
ax.set_xlabel('Radio normalizado $r / r_{max}$')
ax.set_ylabel('Desviación estándar')
ax.set_title('D. Dispersión del colapso')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('fig_radial_universal_collapse.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_radial_universal_collapse.png', dpi=300, bbox_inches='tight')
print("✅ Figura guardada: fig_radial_universal_collapse.pdf/png")

# ==================== RESUMEN ====================
print("\n" + "="*70)
print(" RESUMEN COLAPSO RADIAL")
print("="*70)
print(f"Grafos analizados: {len(profiles)}")
print(f"Varianza media del colapso A(r): {np.mean(std_A):.4f}")
print(f"Varianza media del colapso V(r): {np.mean(std_V):.4f}")

if popt_A is not None:
    print(f"\n📈 Ajuste universal A(r):")
    print(f"   A(r) = {popt_A[0]:.4f} * (r/r_max)^{popt_A[1]:.4f} * (1 - r/r_max)^{popt_A[2]:.4f}")

if popt_V is not None:
    print(f"\n📈 Ajuste universal V(r):")
    print(f"   V(r) = {popt_V[0]:.4f} * (r/r_max)^{popt_V[1]:.4f} * (1 - r/r_max)^{popt_V[2]:.4f}")

if np.mean(std_A) < 0.05:
    print("\n✅ COLAPSO EXCELENTE: geometría radial universal")
elif np.mean(std_A) < 0.1:
    print("\n✅ COLAPSO BUENO: geometría radial reproducible")
else:
    print("\n⚠️ COLAPSO MODERADO: variabilidad entre grafos")

print("\n✅ Script 6 completado.")