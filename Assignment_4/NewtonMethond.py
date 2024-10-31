from typing import Callable

def NewtonMethod_1D(f : Callable[[float], float], df: Callable[[float], float], d2f: Callable[[float], float], epsilon: float, stesize: float, init: float) -> float:
    xk = init
    while df(xk) * df(xk) / d2f(xk) > epsilon:
        xk = xk - stesize * df(xk) / d2f(xk)
    return xk


if __name__ == "__main__":
    f = lambda x: x**3 - x**2 - 1
    df = lambda x: 3*x**2 - 2*x
    d2f = lambda x: 6*x - 2
    optimal_point = NewtonMethod_1D(f, df, d2f, 1e-10, 1e-2, 1)
    print(optimal_point)
    