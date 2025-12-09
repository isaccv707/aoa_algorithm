from aoa import aoa, sphere, rastrigin


def main():
    
    bounds_2d = [(-5.0, 5.0), (-5.0, 5.0)]

    print("===== Ejemplo 1: Sphere 2D =====")
    best_pos, best_score, history = aoa(
        objective_fn=sphere,
        bounds=bounds_2d,
        n_agents=20,
        max_iter=50,
        seed=42,
        verbose=True,
    )

    print("\nResultado final (Sphere 2D):")
    print("  Mejor x:", best_pos)
    print("  Mejor f(x):", best_score)

    
    bounds_3d = [(-5.12, 5.12)] * 3

    print("\n===== Ejemplo 2: Rastrigin 3D =====")
    best_pos2, best_score2, history2 = aoa(
        objective_fn=rastrigin,
        bounds=bounds_3d,
        n_agents=30,
        max_iter=100,
        seed=123,
        verbose=True,
    )

    print("\nResultado final (Rastrigin 3D):")
    print("  Mejor x:", best_pos2)
    print("  Mejor f(x):", best_score2)


if __name__ == "__main__":
    main()
+