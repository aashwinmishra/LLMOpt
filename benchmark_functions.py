import numpy as np


def forrester(x: float) -> float:
    """
    Computes value of the Forrester function, evaluated at x.
    Args:
        x: Input value in [0,1]
    Returns:
        forrester(x)
    """
    return np.square(6 * x - 2) * np.sin(12 * x - 4)
