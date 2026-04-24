import numpy as np
from numba import njit, prange
from typing import Callable, List, Tuple, Optional, Union

# Definimos la lógica de actualización en una función Numba separada para máxima velocidad
@njit(parallel=True)
def _update_population(
    agents, best_pos, mop, moa, range_term, lb, ub, n_agents, n_dims
):
    """
    Actualización vectorizada y paralela de la población usando las reglas de AOA.
    """
    new_agents = np.empty_like(agents)
    
    for i in prange(n_agents):
        for j in range(n_dims):
            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            
            if r1 > moa:
                # --- EXPLORACIÓN ---
                if r2 < 0.5:
                    # op_div = best_pos / (mop + eps) * range_term
                    val = best_pos[j] / (mop + 1e-9) * range_term[j]
                else:
                    # op_mul = best_pos * mop * range_term
                    val = best_pos[j] * mop * range_term[j]
            else:
                # --- EXPLOTACIÓN ---
                if r3 < 0.5:
                    # op_sub = best_pos - mop * range_term
                    val = best_pos[j] - mop * range_term[j]
                else:
                    # op_add = best_pos + mop * range_term
                    val = best_pos[j] + mop * range_term[j]
            
            # Recorte (Clipping)
            if val < lb[j]: val = lb[j]
            if val > ub[j]: val = ub[j]
            new_agents[i, j] = val
            
    return new_agents

def aoa(
    objective_fn: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
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
    if seed is not None:
        np.random.seed(seed)

    n_dims = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # 1) Inicializar población
    agents = np.random.uniform(0, 1, (n_agents, n_dims))
    for j in range(n_dims):
        agents[:, j] = lb[j] + agents[:, j] * (ub[j] - lb[j])
    
    # Fitness inicial
    fitness = np.zeros(n_agents)
    for i in range(n_agents):
        fitness[i] = objective_fn(agents[i])

    # Mejor inicial
    best_idx = np.argmin(fitness)
    best_pos = agents[best_idx].copy()
    best_score = fitness[best_idx]
    best_history = []
    
    range_term = (ub - lb) * mu
    no_improve_counter = 0

    for t in range(1, max_iter + 1):
        # Actualizar Coeficientes
        moa = min_moa + t * (max_moa - min_moa) / max_iter
        mop = 1.0 - (t ** (1.0 / alpha)) / (max_iter ** (1.0 / alpha))

        # Lógica central de actualización (JIT y Paralelizada)
        agents = _update_population(
            agents, best_pos, mop, moa, range_term, lb, ub, n_agents, n_dims
        )

        # Evaluar fitness
        # (Esto sigue siendo el bucle principal; si objective_fn es NJIT, es muy rápido)
        for i in range(n_agents):
            fitness[i] = objective_fn(agents[i])

        # Encontrar el mejor actual
        current_best_idx = np.argmin(fitness)
        current_best_score = fitness[current_best_idx]
        
        if current_best_score < best_score:
            if (best_score - current_best_score) > tol:
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            best_score = current_best_score
            best_pos = agents[current_best_idx].copy()
        else:
            no_improve_counter += 1

        best_history.append(best_score)

        if verbose and t % 10 == 0:
            print(f"Iteración {t:03d} | mejor f(x) = {best_score:.6e}")

        if early_stopping_patience is not None and no_improve_counter >= early_stopping_patience:
            if verbose: print(f"\nParada temprana (early stopping) en la iteración {t}.")
            break

    return best_pos, best_score, best_history