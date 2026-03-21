# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 01:33:31 2026

@author: eggra
"""

# -*- coding: utf-8 -*-
"""
simulate_soup_physical_v20-b.py

VERSIÓN MODIFICADA PARA ESTUDIO DE ESCALADO
- Genera grafos de DIFERENTES TAMAÑOS iniciales
- Guarda resultados para análisis de escalado
- Mantiene toda la funcionalidad original

siguiente->analyze_scaling_vs_N.py
"""

import json
import numpy as np
import random
import os
import warnings
from scipy.sparse import csr_matrix, bmat, save_npz
from scipy.sparse.linalg import eigs
from scipy.stats import linregress
import math
import matplotlib.pyplot as plt
import pandas as pd
import gc

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==================== CONFIGURACIÓN GLOBAL ====================
RANDOM_SEED = 42
STEPS_PER_PARTICLE = 50
INACTIVITY_FACTOR = 5
min_population = 2
output_dir = "soup_simulation_phase_transition_v20"
os.makedirs(output_dir, exist_ok=True)

# ==================== GENERACIÓN DE MATRICES ====================

def generate_adjacency_matrix(N, sigma, omega, IPR_R, seed_offset=0):
    """Genera matriz coherente con propiedades espectrales"""
    random.seed(RANDOM_SEED + seed_offset)
    np.random.seed(RANDOM_SEED + seed_offset)
    
    if N <= 1:
        return csr_matrix((N, N))
    
    # Grafo base Erdos-Renyi
    p = min(0.5, 2.0 / N)
    adj = np.random.random((N, N)) < p
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    
    # Estimar radio espectral
    degrees = np.sum(adj, axis=1)
    mean_degree = np.mean(degrees)
    std_degree = np.std(degrees)
    estimated_radius = mean_degree + 0.5 * std_degree
    
    target_radius = abs(sigma) + abs(omega)
    
    if estimated_radius > 0:
        scale = target_radius / estimated_radius
        adj = (adj * scale).astype(np.float32)
    
    # Añadir antisimetría para omega
    if abs(omega) > 1e-6:
        antisym_strength = 0.2 * abs(omega) / target_radius if target_radius > 0 else 0.1
        antisym = np.random.random((N, N)) < 0.1
        antisym = np.triu(antisym, 1) - np.triu(antisym, 1).T
        adj = adj + antisym * antisym_strength
    
    # Ajustar IPR
    if IPR_R > 0.3:
        star_weight = min(1.0, (IPR_R - 0.3) * 2)
        star = np.zeros((N, N))
        num_connections = max(1, int(N * 0.3))
        connected = random.sample(range(1, N), min(num_connections, N-1))
        for c in connected:
            star[0, c] = 1
            star[c, 0] = 1
        adj = adj * (1 - star_weight) + star * star_weight * target_radius
    
    return csr_matrix(adj)

# ==================== KERNEL REPULSIVO ====================

def bond_energy_repulsive(sigma_a, sigma_b, omega_a, omega_b,
                          Delta_a, Delta_b, IPR_a, IPR_b,
                          k_sigma, k_ipr, k_gap):
    """
    Energía de enlace puramente repulsiva
    E = k_sigma*|Δσ| + k_ipr*(IPR_prod) + k_gap/Δ   (SIEMPRE >0)
    """
    # Protección contra valores no finitos
    if math.isnan(sigma_a) or math.isinf(sigma_a):
        sigma_a = 0.0
    if math.isnan(sigma_b) or math.isinf(sigma_b):
        sigma_b = 0.0
    if math.isnan(omega_a) or math.isinf(omega_a):
        omega_a = 0.0
    if math.isnan(omega_b) or math.isinf(omega_b):
        omega_b = 0.0
    if math.isnan(Delta_a) or Delta_a <= 0:
        Delta_a = 0.1
    if math.isnan(Delta_b) or Delta_b <= 0:
        Delta_b = 0.1
    if math.isnan(IPR_a):
        IPR_a = 1.0
    if math.isnan(IPR_b):
        IPR_b = 1.0

    delta_sigma = abs(sigma_a - sigma_b)
    ipr_product = IPR_a * IPR_b
    geom_gap = math.sqrt(Delta_a * Delta_b)
    if geom_gap <= 0:
        geom_gap = 0.001
    
    # TODOS los términos positivos → energía siempre positiva
    E = k_sigma * delta_sigma + k_ipr * ipr_product + k_gap / geom_gap
    
    # Limitar para evitar valores extremos
    if E > 100:
        E = 100
    elif E < -100:
        E = -100
    
    return E

# ==================== FUNCIONES CON MATRICES DISPERSAS ====================

def build_combined_sparse(adj1, adj2, coupling_strength, n1, n2):
    """
    Construye matriz combinada dispersa
    Permite 0 conexiones si coupling_strength es pequeño
    """
    max_possible = n1 * n2
    target_connections = int(coupling_strength * min(n1, n2))
    num_connections = min(target_connections, max_possible // 2)
    
    # Si num_connections = 0, no se añaden conexiones
    if num_connections == 0:
        return bmat([[adj1, None], [None, adj2]], format='csr')
    
    rows = []
    cols = []
    data = []
    
    n1_samples = min(num_connections, n1)
    n2_samples = min(num_connections, n2)
    
    # Conexiones 1→2
    sources1 = random.sample(range(n1), n1_samples)
    targets2 = random.sample(range(n2), n2_samples)
    
    for i in range(num_connections):
        s = sources1[i % n1_samples]
        t = targets2[i % n2_samples]
        rows.append(s)
        cols.append(n1 + t)
        data.append(1)
    
    # Conexiones 2→1
    sources2 = random.sample(range(n2), n2_samples)
    targets1 = random.sample(range(n1), n1_samples)
    
    for i in range(num_connections):
        s = sources2[i % n2_samples]
        t = targets1[i % n1_samples]
        rows.append(n1 + s)
        cols.append(t)
        data.append(1)
    
    connections = csr_matrix((data, (rows, cols)), shape=(n1+n2, n1+n2))
    combined = bmat([[adj1, None], [None, adj2]], format='csr')
    combined = combined + connections
    
    return combined

def compute_spectral_properties_sparse(adj_matrix):
    """Calcula propiedades espectrales"""
    N = adj_matrix.shape[0]
    
    result = {
        "sigma": 0.0,
        "omega": 0.0,
        "Delta": 0.1,
        "IPR_R": 1.0,
        "eigenvector_real": True,
        "spectral_radius": 0.0
    }
    
    if N < 2:
        return result
    
    try:
        k = min(5, N-1)
        if k > 0:
            eigenvalues, eigenvectors = eigs(adj_matrix, k=k, which='LM', 
                                            maxiter=20*N, tol=1e-4)
            
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            lambda_max = eigenvalues[0]
            vR = eigenvectors[:, 0]
            
            result["spectral_radius"] = float(np.abs(lambda_max))
            
            vR_norm = vR / np.linalg.norm(vR)
            vR_real = np.real(vR_norm)
            IPR_R = np.sum(vR_real**4)
            result["IPR_R"] = float(IPR_R) if not np.isnan(IPR_R) else 1.0
            
            result["sigma"] = float(lambda_max.real)
            result["omega"] = float(lambda_max.imag)
            
            if len(eigenvalues) > 1:
                gap = np.abs(eigenvalues[0]) - np.abs(eigenvalues[1])
                result["Delta"] = float(max(0.001, abs(gap)))
            
    except Exception as e:
        pass
    
    return result

# ==================== CLASE MOLÉCULA ====================

class Molecule:
    __slots__ = ['id', 'N', 'generation', 'sigma', 'omega', 'Delta', 'IPR_R', 
                 'adj_matrix', 'components', 'history', 'formation_step']
    
    def __init__(self, id, N, generation, sigma, omega, Delta, IPR_R, 
                 adj_matrix, components=None, history=None, formation_step=None):
        self.id = id
        self.N = N
        self.generation = generation
        self.sigma = sigma
        self.omega = omega
        self.Delta = Delta
        self.IPR_R = IPR_R
        self.adj_matrix = adj_matrix
        self.components = components or []
        self.history = history or [id]
        self.formation_step = formation_step

# ==================== FUNCIÓN PRINCIPAL DE SIMULACIÓN ====================

def run_simulation(coupling_strength, k_sigma, k_ipr, k_gap, T, seed, initial_atoms=None):
    """
    Ejecuta simulación con kernel repulsivo
    Ahora acepta initial_atoms para usar diferentes bases de datos
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Cargar base de datos (por defecto o la especificada)
    if initial_atoms is None:
        with open("atom_database.json") as f:
            atoms_data = json.load(f)
    else:
        atoms_data = initial_atoms
    
    initial_population = len(atoms_data)
    max_steps = STEPS_PER_PARTICLE * initial_population
    inactivity_limit = INACTIVITY_FACTOR * initial_population
    
    print(f"   📊 N0={initial_population}, max_steps={max_steps}, inactivity={inactivity_limit}")
    
    # Crear moléculas iniciales
    population = []
    for i, atom_data in enumerate(atoms_data):
        N = atom_data["N"]
        
        adj_matrix = generate_adjacency_matrix(
            N=N,
            sigma=atom_data["sigma"],
            omega=atom_data.get("omega", 0),
            IPR_R=atom_data.get("IPR_R", 1.0),
            seed_offset=i
        )
        
        mol = Molecule(
            id=i,
            N=N,
            generation=0,
            sigma=float(atom_data.get("sigma", 0.0) or 0.0),
            omega=float(atom_data.get("omega", 0.0) or 0.0),
            Delta=float(atom_data.get("Delta", 0.1) or 0.1),
            IPR_R=float(atom_data.get("IPR_R", 1.0) or 1.0),
            adj_matrix=adj_matrix,
            components=[],
            history=[i],
            formation_step=0
        )
        population.append(mol)
    
    print(f"   ✅ {len(population)} moléculas generadas")
    
    next_id = len(population)
    fusion_count = 0
    steps_without_fusion = 0
    
    fusion_sizes = []
    energy_history = []
    attempted_fusions = 0
    
    for step in range(max_steps):
        if len(population) < 2:
            break
        
        idx_a, idx_b = random.sample(range(len(population)), 2)
        mol_a = population[idx_a]
        mol_b = population[idx_b]
        
        new_generation = max(mol_a.generation, mol_b.generation) + 1
        
        # Calcular energía con kernel repulsivo
        E = bond_energy_repulsive(
            mol_a.sigma, mol_b.sigma,
            mol_a.omega, mol_b.omega,
            mol_a.Delta, mol_b.Delta,
            mol_a.IPR_R, mol_b.IPR_R,
            k_sigma, k_ipr, k_gap
        )
        
        energy_history.append(E)
        attempted_fusions += 1
        
        # Probabilidad logística con protección contra overflow
        try:
            exp_arg = min(50.0, E / T)
            P = 1.0 / (1.0 + math.exp(exp_arg))
            if np.isinf(P) or np.isnan(P):
                P = 0.0
        except:
            P = 0.0
        
        if random.random() < P:
            combined_adj = build_combined_sparse(
                mol_a.adj_matrix, mol_b.adj_matrix, 
                coupling_strength, mol_a.N, mol_b.N
            )
            
            spectral = compute_spectral_properties_sparse(combined_adj)
            
            new_mol = Molecule(
                id=next_id,
                N=mol_a.N + mol_b.N,
                generation=new_generation,
                sigma=spectral["sigma"],
                omega=spectral["omega"],
                Delta=spectral["Delta"],
                IPR_R=spectral["IPR_R"],
                adj_matrix=combined_adj,
                components=[mol_a.id, mol_b.id],
                history=mol_a.history + mol_b.history + [next_id],
                formation_step=step
            )
            next_id += 1
            fusion_count += 1
            fusion_sizes.append(new_mol.N)
            
            if idx_a < idx_b:
                population.pop(idx_b)
                population.pop(idx_a)
            else:
                population.pop(idx_a)
                population.pop(idx_b)
            
            population.append(new_mol)
            steps_without_fusion = 0
            
            if fusion_count % 100 == 0:
                gc.collect()
        else:
            steps_without_fusion += 1
        
        if steps_without_fusion >= inactivity_limit:
            break
    
    # ========== ANÁLISIS FINAL ==========
    sizes = [m.N for m in population]
    total_nodes = sum(sizes)
    max_size = max(sizes) if sizes else 0
    mean_size = np.mean(sizes) if sizes else 0
    fraction_largest = max_size / total_nodes if total_nodes > 0 else 0
    
    population_size = len(sizes)
    
    # ========== GUARDAR GRAFO GIGANTE ==========
    if population_size == 1:  # Solo si hay un cluster
        largest_mol = max(population, key=lambda m: m.N)
        giant_adj = largest_mol.adj_matrix
        
        # Crear nombre de archivo con parámetros
        filename = f"giant_graph_k{int(k_sigma)}_{int(k_ipr)}_{int(k_gap)}_T{int(T)}_l{int(coupling_strength*100):03d}_Ninit{initial_population}.npz"
        filepath = os.path.join(output_dir, filename)
        
        save_npz(filepath, giant_adj)
        print(f"   💾 Grafo gigante guardado: {giant_adj.shape} en {filename}")
        
        # Guardar también información de escalado
        scaling_info = {
            'N0': initial_population,
            'N_final': giant_adj.shape[0],
            'coupling': coupling_strength,
            'k_sigma': k_sigma,
            'k_ipr': k_ipr,
            'k_gap': k_gap,
            'T': T,
            'seed': seed,
            'filename': filename
        }
        return scaling_info
    else:
        print(f"   ⚠️ No hay un único cluster final (población={population_size})")
        return None

# ==================== EXPERIMENTO PRINCIPAL MODIFICADO ====================

print("\n" + "="*70)
print(" ESTUDIO DE ESCALADO - VERSIÓN MODIFICADA")
print("="*70)
print(" ✓ Genera grafos de DIFERENTES TAMAÑOS iniciales")
print(" ✓ Para estudiar escalado: D(N) vs N")
print(" ✓ Mantiene toda la funcionalidad original")

# Parámetros fijos
k_sigma = 5.0
k_ipr = 5.0
k_gap = 10.0
T = 100.0
coupling = 0.2  # λ fijo para el estudio de escalado

# DIFERENTES TAMAÑOS INICIALES (N0)
initial_sizes = [200, 500, 1000, 2000, 5000, 10000, 20000]
replicas = 3  # 3 réplicas por tamaño

print(f"\n🔬 PARÁMETROS FIJOS PARA ESCALADO:")
print(f"   k_σ = {k_sigma}")
print(f"   k_ipr = {k_ipr}")
print(f"   k_gap = {k_gap}")
print(f"   T = {T}")
print(f"   λ = {coupling} (fijo)")
print(f"   Tamaños iniciales: {initial_sizes}")
print(f"   Réplicas por tamaño: {replicas}")

# Cargar base de datos de átomos una sola vez
with open("atom_database.json") as f:
    full_atoms_data = json.load(f)

scaling_results = []

for N0 in initial_sizes:
    print(f"\n📊 TAMAÑO INICIAL = {N0}")
    
    # Tomar una muestra aleatoria de átomos de tamaño N0
    atoms_sample = random.sample(full_atoms_data, min(N0, len(full_atoms_data)))
    
    for r in range(replicas):
        seed = RANDOM_SEED + r + N0 * 10  # Semilla diferente para cada tamaño
        print(f"   Réplica {r+1}/{replicas}...")
        
        gc.collect()
        
        result = run_simulation(
            coupling_strength=coupling,
            k_sigma=k_sigma,
            k_ipr=k_ipr,
            k_gap=k_gap,
            T=T,
            seed=seed,
            initial_atoms=atoms_sample
        )
        
        if result is not None:
            scaling_results.append(result)
            print(f"      → N_final = {result['N_final']}")
            
            # Guardado parcial
            partial_df = pd.DataFrame(scaling_results)
            partial_df.to_csv(os.path.join(output_dir, "scaling_results_partial.csv"), index=False)

# ==================== GUARDAR RESULTADOS FINALES ====================
if scaling_results:
    df_scaling = pd.DataFrame(scaling_results)
    csv_path = os.path.join(output_dir, "scaling_results_final.csv")
    df_scaling.to_csv(csv_path, index=False)
    
    print("\n" + "="*70)
    print(" RESULTADOS DEL ESTUDIO DE ESCALADO")
    print("="*70)
    print(df_scaling.to_string())
    
    # Análisis rápido
    print("\n" + "="*70)
    print(" ANÁLISIS PRELIMINAR")
    print("="*70)
    
    for N0 in initial_sizes:
        data = df_scaling[df_scaling['N0'] == N0]
        if len(data) > 0:
            print(f"\nN0 = {N0}:")
            print(f"   N_final medio = {data['N_final'].mean():.0f} ± {data['N_final'].std():.0f}")
    
    print("\n" + "="*70)
    print(" SIGUIENTE PASO:")
    print("="*70)
    print("""
    Ejecuta ahora el análisis de escalado:
    
    python analyze_scaling_final.py
    
    Esto generará las gráficas y determinará si:
    - D ~ log(N)  (small-world)
    - D ~ N^α     (fractal)
    """)
    
else:
    print("\n❌ No se obtuvieron resultados")

print(f"\n✅ Estudio completado. Resultados en: {output_dir}/scaling_results_final.csv")