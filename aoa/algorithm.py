import numpy as np
from numba import njit, prange
from typing import Callable, List, Tuple, Optional, Union

# Define the update logic in a separate Numba function for maximum speed
@njit(parallel=True)
def _update_population(
    agents, best_pos, mop, moa, range_term, lb, ub, n_agents, n_dims
):
    """
    Vectorized and Parallelized update of the population using AOA rules.
    """
    new_agents = np.empty_like(agents)
    
    for i in prange(n_agents):
        for j in range(n_dims):
            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            
            if r1 > moa:
                # --- EXPLORATION ---
                if r2 < 0.5:
                    # op_div = best_pos / (mop + eps) * range_term
                    val = best_pos[j] / (mop + 1e-9) * range_term[j]
                else:
                    # op_mul = best_pos * mop * range_term
                    val = best_pos[j] * mop * range_term[j]
            else:
                # --- EXPLOITATION ---
                if r3 < 0.5:
                    # op_sub = best_pos - mop * range_term
                    val = best_pos[j] - mop * range_term[j]
                else:
                    # op_add = best_pos + mop * range_term
                    val = best_pos[j] + mop * range_term[j]
            
            # Clipping
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

    # 1) Initialize population
    agents = np.random.uniform(0, 1, (n_agents, n_dims))
    for j in range(n_dims):
        agents[:, j] = lb[j] + agents[:, j] * (ub[j] - lb[j])
    
    # Initial fitness
    fitness = np.zeros(n_agents)
    for i in range(n_agents):
        fitness[i] = objective_fn(agents[i])

    # Initial best
    best_idx = np.argmin(fitness)
    best_pos = agents[best_idx].copy()
    best_score = fitness[best_idx]
    best_history = []
    
    range_term = (ub - lb) * mu
    no_improve_counter = 0

    for t in range(1, max_iter + 1):
        # Update Coefficients
        moa = min_moa + t * (max_moa - min_moa) / max_iter
        mop = 1.0 - (t ** (1.0 / alpha)) / (max_iter ** (1.0 / alpha))

        # Core update logic (JIT and Parallelized)
        agents = _update_population(
            agents, best_pos, mop, moa, range_term, lb, ub, n_agents, n_dims
        )

        # Evaluate fitness
        # (This is still the main loop; if objective_fn is NJIT, this is very fast)
        for i in range(n_agents):
            fitness[i] = objective_fn(agents[i])

        # Find current best
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
            if verbose: print(f"\nEarly stopping en {t}.")
            break

    return best_pos, best_score, best_history
