from typing import List

Vector = List[float]


def sphere(x: Vector) -> float:
    """
    Sphere function: f(x) = sum(x_i^2)
    Mínimo en x = 0...0 con f(x) = 0
    """
    return sum(v ** 2 for v in x)


def rastrigin(x: Vector) -> float:
    """
    Rastrigin function (clásica para pruebas de optimización):

    f(x) = 10 * d + sum[x_i^2 - 10 * cos(2*pi*x_i)]

    Tiene muchos mínimos locales, el global está en x = 0...0
    """
    import math

    d = len(x)
    return 10 * d + sum(v ** 2 - 10 * math.cos(2 * math.pi * v) for v in x)
