import numpy as np
from typing import Callable, List, Tuple, Optional, Union

# Definiciones de tipos para mayor claridad
Vector = Union[List[float], np.ndarray]
Bounds = List[Tuple[float, float]]

def aoa(
    objective_fn: Callable[[np.ndarray], float],
    bounds: Bounds,
    n_agents: int = 30,
    max_iter: int = 100,
    alpha: float = 5.0,
    mu: float = 0.5,
    min_moa: float = 0.2,
    max_moa: float = 0.9,
    seed: Optional[int] = None,
    verbose: bool = True,
    early_stopping_patience: Optional[int] = None,
    tol: float = 1e-8,
):
    """
    Versión vectorizada del Arithmetic Optimization Algorithm (AOA) usando NumPy.
    Optimiza el rendimiento eliminando bucles internos para agentes y dimensiones.
    """
    if seed is not None:
        np.random.seed(seed)

    n_dims = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # 1) Inicializar población (Vectorizado)
    agents = np.random.uniform(lb, ub, (n_agents, n_dims))
    
    # Evaluar población inicial
    fitness = np.array([objective_fn(agent) for agent in agents])

    # Encontrar mejor agente inicial
    best_idx = np.argmin(fitness)
    best_pos = agents[best_idx].copy()
    best_score = fitness[best_idx]

    best_history = []
    
    # Precalcular término de rango por dimensión
    range_term = (ub - lb) * mu
    
    # Variables para early stopping
    no_improve_counter = 0

    # Bucle principal
    for t in range(1, max_iter + 1):
        # 2) Actualizar MOA y MOP (Escalares)
        moa = min_moa + t * (max_moa - min_moa) / max_iter
        mop = 1.0 - (t ** (1.0 / alpha)) / (max_iter ** (1.0 / alpha))

        # Generar matrices de números aleatorios para toda la población
        # Forma: (n_agents, n_dims)
        R1 = np.random.random((n_agents, n_dims))
        R2 = np.random.random((n_agents, n_dims))
        R3 = np.random.random((n_agents, n_dims))

        # 3) Cálculo de candidatos para cada operador (Vectorizado)
        # El broadcasting de NumPy aplica best_pos y range_term a cada fila (agente)
        
        # --- Fase de EXPLORACIÓN ---
        op_div = best_pos / (mop + 1e-9) * range_term
        op_mul = best_pos * mop * range_term
        explor_val = np.where(R2 < 0.5, op_div, op_mul)

        # --- Fase de EXPLOTACIÓN ---
        op_sub = best_pos - mop * range_term
        op_add = best_pos + mop * range_term
        exploit_val = np.where(R3 < 0.5, op_sub, op_add)

        # Decisión final por elemento: Exploración si R1 > moa, de lo contrario Explotación
        new_agents = np.where(R1 > moa, explor_val, exploit_val)

        # 4) Limitar al rango y Evaluar (Casting implícito en la evaluación)
        agents = np.clip(new_agents, lb, ub)
        fitness = np.array([objective_fn(agent) for agent in agents])

        # Buscar nuevo mejor
        current_best_idx = np.argmin(fitness)
        current_best_score = fitness[current_best_idx]
        
        prev_best_score = best_score
        if current_best_score < best_score:
            best_score = current_best_score
            best_pos = agents[current_best_idx].copy()
            
            # Verificar si la mejora es significativa para resetear early stopping
            if (prev_best_score - best_score) > tol:
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1

        best_history.append(best_score)

        if verbose:
            print(f"Iteración {t:03d} | mejor f(x) = {best_score:.6e}")

        # 5) Early stopping
        if early_stopping_patience is not None:
            if no_improve_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping activado en la iteración {t}.")
                break

    return best_pos, best_score, best_history
