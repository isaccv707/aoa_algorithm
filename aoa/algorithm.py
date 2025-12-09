import random
from typing import Callable, List, Tuple, Optional


Vector = List[float]
Bounds = List[Tuple[float, float]]  # [(lb1, ub1), (lb2, ub2), ...]


def aoa(
    objective_fn: Callable[[Vector], float],
    bounds: Bounds,
    n_agents: int = 30,
    max_iter: int = 100,
    alpha: float = 5.0,
    mu: float = 0.5,
    min_moa: float = 0.2,
    max_moa: float = 0.9,
    seed: Optional[int] = None,
    verbose: bool = True,
):
    """
    Implementación sencilla y genérica del Arithmetic Optimization Algorithm (AOA).

    - objective_fn: función objetivo que recibe una lista [x1, x2, ..., xd] y regresa un float.
    - bounds: lista de (min, max) por cada dimensión.
    - n_agents: tamaño de la población.
    - max_iter: número máximo de iteraciones.
    """

    if seed is not None:
        random.seed(seed)

    n_dims = len(bounds)

    # 1) Inicializar población
    agents: List[Vector] = []
    for _ in range(n_agents):
        agent = [
            random.uniform(lb, ub)
            for (lb, ub) in bounds
        ]
        agents.append(agent)

    # Evaluar población inicial
    fitness = [objective_fn(agent) for agent in agents]

    # Encontrar mejor agente inicial
    best_index = min(range(n_agents), key=lambda i: fitness[i])
    best_pos: Vector = agents[best_index][:]
    best_score: float = fitness[best_index]

    best_history: List[float] = []

    # Precalcular término de rango por dimensión
    # IMPORTANTE: solo usamos el tamaño del rango, sin sumar lb
    # range_term[j] = (UB_j - LB_j) * mu
    range_term: Vector = []
    for lb, ub in bounds:
        range_term.append((ub - lb) * mu)

    # Bucle principal
    for t in range(1, max_iter + 1):
        # 2) Actualizar MOA y MOP
        moa = min_moa + t * (max_moa - min_moa) / max_iter
        mop = 1.0 - (t ** (1.0 / alpha)) / (max_iter ** (1.0 / alpha))

        new_agents: List[Vector] = []

        # 3) Actualizar cada agente
        for i in range(n_agents):
            current = agents[i]
            new_position: Vector = []

            for j in range(n_dims):
                lb, ub = bounds[j]

                r1 = random.random()
                r2 = random.random()
                r3 = random.random()

                if r1 > moa:
                    # EXPLORACIÓN (Multiplicación / División)
                    if r2 < 0.5:
                        # División
                        new_x = best_pos[j] / (mop + 1e-9) * range_term[j]
                    else:
                        # Multiplicación
                        new_x = best_pos[j] * mop * range_term[j]
                else:
                    # EXPLOTACIÓN (Suma / Resta)
                    if r3 < 0.5:
                        # Resta
                        new_x = best_pos[j] - mop * range_term[j]
                    else:
                        # Suma
                        new_x = best_pos[j] + mop * range_term[j]

                # 4) Limitar al rango [lb, ub]
                if new_x < lb:
                    new_x = lb
                elif new_x > ub:
                    new_x = ub

                new_position.append(new_x)

            new_agents.append(new_position)

        # Reemplazamos población
        agents = new_agents
        fitness = [objective_fn(agent) for agent in agents]

        # Buscar nuevo mejor
        best_index = min(range(n_agents), key=lambda i: fitness[i])
        if fitness[best_index] < best_score:
            best_score = fitness[best_index]
            best_pos = agents[best_index][:]

        best_history.append(best_score)

        if verbose:
            print(
                f"Iteración {t:03d} | mejor f(x) = {best_score:.6f} | "
                f"mejor x = {[round(v, 4) for v in best_pos]}"
            )

    return best_pos, best_score, best_history
