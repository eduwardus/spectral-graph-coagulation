# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:56:39 2026

@author: eggra
"""

# -*- coding: utf-8 -*-
"""
simulate_soup_physical_v20-e.py

VERSIÓN CON VARIACIÓN DE ENLACES POR FUSIÓN
- Prueba k = 2, 5, 10 enlaces por fusión
- Para verificar: k_core_max ≈ floor(⟨k⟩)
->analize_kcore_structure.py
->experimento_A_variar_parametros.py
"""

import json
import numpy as np
import random
import os
import warnings
from scipy.sparse import csr_matrix, bmat, save_npz
from scipy.sparse.linalg import eigs
import math
import pandas as pd
import gc
from datetime import datetime
import pickle

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==================== CONFIGURACIÓN GLOBAL ====================
RANDOM_SEED = 42
STEPS_PER_PARTICLE = 50
INACTIVITY_FACTOR = 5
min_population = 2
output_dir = "soup_simulation_phase_transition_v20"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "snapshots"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "population"), exist_ok=True)

# ==================== GENERACIÓN DE MATRICES ====================

def generate_adjacency_matrix(N, sigma, omega, IPR_R, seed_offset=0):
    """Genera matriz coherente con propiedades espectrales"""
    random.seed(RANDOM_SEED + seed_offset)
    np.random.seed(RANDOM_SEED + seed_offset)
    
    if N <= 1:
        return csr_matrix((N, N))
    
    p = min(0.5, 2.0 / N)
    adj = np.random.random((N, N)) < p
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    
    degrees = np.sum(adj, axis=1)
    mean_degree = np.mean(degrees)
    std_degree = np.std(degrees)
    estimated_radius = mean_degree + 0.5 * std_degree
    
    target_radius = abs(sigma) + abs(omega)
    
    if estimated_radius > 0:
        scale = target_radius / estimated_radius
        adj = (adj * scale).astype(np.float32)
    
    if abs(omega) > 1e-6:
        antisym_strength = 0.2 * abs(omega) / target_radius if target_radius > 0 else 0.1
        antisym = np.random.random((N, N)) < 0.1
        antisym = np.triu(antisym, 1) - np.triu(antisym, 1).T
        adj = adj + antisym * antisym_strength
    
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
    """Energía de enlace puramente repulsiva"""
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
    
    E = k_sigma * delta_sigma + k_ipr * ipr_product + k_gap / geom_gap
    
    if E > 100:
        E = 100
    elif E < -100:
        E = -100
    
    return E

# ==================== FUNCIÓN PARA GUARDAR POBLACIÓN COMPLETA ====================

def save_population_state(population, step, fusion_count, sim_id, params):
    """Guarda el estado completo de la población"""
    if len(population) == 0:
        return
    
    sizes = [mol.N for mol in population]
    
    state = {
        'sim_id': sim_id,
        'step': step,
        'fusion_count': fusion_count,
        'n_clusters': len(sizes),
        'sizes': sizes,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"population_fusion{fusion_count:06d}_step{step:06d}_{params}_sim{sim_id}.pkl"
    filepath = os.path.join(output_dir, "population", filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

# ==================== FUNCIÓN PARA GUARDAR SNAPSHOTS ====================

def save_snapshot(mol, step, fusion_count, output_dir, params, sim_id):
    """Guarda un snapshot del grafo gigante"""
    if mol.N < 100:
        return
    
    filename = f"snapshot_fusion{fusion_count:06d}_step{step:06d}_N{mol.N}_{params}_sim{sim_id}.npz"
    filepath = os.path.join(output_dir, "snapshots", filename)
    
    save_npz(filepath, mol.adj_matrix)
    print(f"      📸 Snapshot: fusion={fusion_count}, N={mol.N}")

# ==================== FUNCIÓN PARA GUARDAR EVENTO DE FUSIÓN ====================

def save_fusion_event(sim_id, fusion_id, size_a, size_b, size_new, params, k_links):
    """Guarda un evento de fusión"""
    fusion_history_file = os.path.join(output_dir, f"fusion_history_k{k_links}.csv")
    
    event = {
        'timestamp': datetime.now().isoformat(),
        'sim_id': sim_id,
        'fusion_id': fusion_id,
        'size_a': size_a,
        'size_b': size_b,
        'size_new': size_new,
        'k_links': k_links,
        'coupling': params['coupling'],
        'k_sigma': params['k_sigma'],
        'k_ipr': params['k_ipr'],
        'k_gap': params['k_gap'],
        'T': params['T']
    }
    
    df_event = pd.DataFrame([event])
    if not os.path.exists(fusion_history_file):
        df_event.to_csv(fusion_history_file, index=False)
    else:
        df_event.to_csv(fusion_history_file, mode='a', header=False, index=False)

# ==================== FUNCIÓN MODIFICADA CON NÚMERO DE ENLACES VARIABLE ====================

def build_combined_sparse(adj1, adj2, coupling_strength, n1, n2, k_links):
    """
    Construye matriz combinada dispersa
    Ahora con parámetro k_links = número de enlaces por fusión
    """
    max_possible = n1 * n2
    target_connections = int(coupling_strength * min(n1, n2))
    # Usar k_links como número base de conexiones
    num_connections = min(k_links, max_possible // 2)
    
    if num_connections == 0:
        return bmat([[adj1, None], [None, adj2]], format='csr')
    
    rows = []
    cols = []
    data = []
    
    n1_samples = min(num_connections, n1)
    n2_samples = min(num_connections, n2)
    
    sources1 = random.sample(range(n1), n1_samples)
    targets2 = random.sample(range(n2), n2_samples)
    
    for i in range(num_connections):
        s = sources1[i % n1_samples]
        t = targets2[i % n2_samples]
        rows.append(s)
        cols.append(n1 + t)
        data.append(1)
    
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

# ==================== FUNCIÓN PRINCIPAL MODIFICADA ====================

def run_simulation(coupling_strength, k_sigma, k_ipr, k_gap, T, seed, 
                  initial_atoms=None, sim_id=0, k_links=5):
    """
    Ejecuta simulación con número variable de enlaces por fusión
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if initial_atoms is None:
        with open("atom_database.json") as f:
            atoms_data = json.load(f)
    else:
        atoms_data = initial_atoms
    
    initial_population = len(atoms_data)
    max_steps = STEPS_PER_PARTICLE * initial_population
    inactivity_limit = INACTIVITY_FACTOR * initial_population
    
    print(f"   📊 N0={initial_population}, k_links={k_links}")
    
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
    
    params_str = f"k{int(k_sigma)}_{int(k_ipr)}_{int(k_gap)}_T{int(T)}_l{int(coupling_strength*100):03d}_links{k_links}"
    SNAPSHOT_FREQUENCY = 50
    
    fusion_params = {
        'coupling': coupling_strength,
        'k_sigma': k_sigma,
        'k_ipr': k_ipr,
        'k_gap': k_gap,
        'T': T
    }
    
    # Guardar población inicial
    save_population_state(population, 0, fusion_count, sim_id, params_str)
    
    for step in range(max_steps):
        if len(population) < 2:
            break
        
        idx_a, idx_b = random.sample(range(len(population)), 2)
        mol_a = population[idx_a]
        mol_b = population[idx_b]
        
        new_generation = max(mol_a.generation, mol_b.generation) + 1
        
        E = bond_energy_repulsive(
            mol_a.sigma, mol_b.sigma,
            mol_a.omega, mol_b.omega,
            mol_a.Delta, mol_b.Delta,
            mol_a.IPR_R, mol_b.IPR_R,
            k_sigma, k_ipr, k_gap
        )
        
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
                coupling_strength, mol_a.N, mol_b.N, k_links
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
            
            # Guardar evento
            save_fusion_event(sim_id, fusion_count, mol_a.N, mol_b.N, new_mol.N, fusion_params, k_links)
            
            # Snapshots
            if fusion_count > 0 and fusion_count % SNAPSHOT_FREQUENCY == 0:
                save_snapshot(new_mol, step, fusion_count, output_dir, params_str, sim_id)
                save_population_state(population, step, fusion_count, sim_id, params_str)
            
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
    
    # Guardar población final
    save_population_state(population, step, fusion_count, sim_id, params_str)
    
    if len(population) == 1:
        largest_mol = max(population, key=lambda m: m.N)
        giant_adj = largest_mol.adj_matrix
        
        filename = f"giant_graph_k{int(k_sigma)}_{int(k_ipr)}_{int(k_gap)}_T{int(T)}_l{int(coupling_strength*100):03d}_Ninit{initial_population}_links{k_links}_sim{sim_id}.npz"
        filepath = os.path.join(output_dir, filename)
        
        save_npz(filepath, giant_adj)
        print(f"   💾 Grafo gigante: {giant_adj.shape}")
        
        return {
            'N0': initial_population,
            'N_final': giant_adj.shape[0],
            'k_links': k_links,
            'sim_id': sim_id,
            'fusion_count': fusion_count
        }
    return None

# ==================== EXPERIMENTO PRINCIPAL ====================

print("\n" + "="*70)
print(" ESTUDIO DE VARIACIÓN DE ENLACES POR FUSIÓN")
print("="*70)
print(" Probando k_links = 2, 5, 10")

# Parámetros fijos
k_sigma = 5.0
k_ipr = 5.0
k_gap = 10.0
T = 100.0
coupling = 0.2

# Valores a probar
k_links_values = [2, 5, 10]
initial_sizes = [1000]  # Un tamaño fijo para comparar
replicas = 3

print(f"\n🔬 PARÁMETROS:")
print(f"   k_σ = {k_sigma}")
print(f"   k_ipr = {k_ipr}")
print(f"   k_gap = {k_gap}")
print(f"   T = {T}")
print(f"   λ = {coupling}")
print(f"   Tamaño fijo: {initial_sizes[0]}")
print(f"   Réplicas: {replicas}")

with open("atom_database.json") as f:
    full_atoms_data = json.load(f)

all_results = []
sim_counter = 0

for k_links in k_links_values:
    print(f"\n{'='*50}")
    print(f"📊 EXPERIMENTO: k_links = {k_links}")
    print(f"{'='*50}")
    
    atoms_sample = random.sample(full_atoms_data, min(initial_sizes[0], len(full_atoms_data)))
    
    for r in range(replicas):
        seed = RANDOM_SEED + r + k_links * 100
        print(f"\n   Réplica {r+1}/{replicas} (sim_id={sim_counter})...")
        
        gc.collect()
        
        result = run_simulation(
            coupling_strength=coupling,
            k_sigma=k_sigma,
            k_ipr=k_ipr,
            k_gap=k_gap,
            T=T,
            seed=seed,
            initial_atoms=atoms_sample,
            sim_id=sim_counter,
            k_links=k_links
        )
        
        if result is not None:
            all_results.append(result)
            print(f"      → N_final = {result['N_final']}, fusiones = {result['fusion_count']}")
        
        sim_counter += 1

# ==================== GUARDAR RESULTADOS ====================
if all_results:
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, "klinks_experiment_results.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*70)
    print(" RESULTADOS FINALES")
    print("="*70)
    
    for k_links in k_links_values:
        data = df[df['k_links'] == k_links]
        if len(data) > 0:
            print(f"\nk_links = {k_links}:")
            print(f"   N_final medio = {data['N_final'].mean():.0f} ± {data['N_final'].std():.0f}")
            print(f"   Fusiones medias = {data['fusion_count'].mean():.0f} ± {data['fusion_count'].std():.0f}")
    
    print("\n" + "="*70)
    print(" SIGUIENTE PASO:")
    print("="*70)
    print("""
    Ejecuta el análisis de k-core sobre estos nuevos grafos:
    
    python analyze_kcore_structure_detailed.py
    
    Verificarás si k_core_max ≈ floor(⟨k⟩) se mantiene
    para diferentes valores de k_links.
    """)
    
else:
    print("\n❌ No se obtuvieron resultados")

print(f"\n✅ Estudio completado. Resultados en: {output_dir}/")