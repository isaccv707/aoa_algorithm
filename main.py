from aoa import aoa, sphere, rastrigin
import matplotlib.pyplot as plt


def plot_convergence(history, title: str):
    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteración")
    plt.ylabel("Mejor f(x)")
    plt.title(title)
    plt.grid(True)
    # Para problemas donde f(x) baja mucho, a veces ayuda:
    # plt.yscale("log")
    plt.show()


def main():
    # Ejemplo 1: minimizar Sphere en 2D en [-5, 5]
    bounds_2d = [(-5.0, 5.0), (-5.0, 5.0)]

    print("===== Ejemplo 1: Sphere 2D =====")
    best_pos, best_score, history = aoa(
        objective_fn=sphere,
        bounds=bounds_2d,
        n_agents=20,
        max_iter=200,
        seed=42,
        verbose=True,
        early_stopping_patience=30,  # si 30 iteraciones sin mejora, paramos
        tol=1e-10,
    )

    print("\nResultado final (Sphere 2D):")
    print("  Mejor x:", best_pos)
    print("  Mejor f(x):", best_score)

    # Gráfica de convergencia para Sphere
    plot_convergence(history, "Convergencia - Sphere 2D")

    # Ejemplo 2: minimizar Rastrigin en 3D en [-5.12, 5.12]
    bounds_3d = [(-5.12, 5.12)] * 3

    print("\n===== Ejemplo 2: Rastrigin 3D =====")
    best_pos2, best_score2, history2 = aoa(
        objective_fn=rastrigin,
        bounds=bounds_3d,
        n_agents=30,
        max_iter=500,
        seed=123,
        verbose=True,
        early_stopping_patience=50,
        tol=1e-8,
    )

    print("\nResultado final (Rastrigin 3D):")
    print("  Mejor x:", best_pos2)
    print("  Mejor f(x):", best_score2)

    # Gráfica de convergencia para Rastrigin
    plot_convergence(history2, "Convergencia - Rastrigin 3D")


if __name__ == "__main__":
    main()
