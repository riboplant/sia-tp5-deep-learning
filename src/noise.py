import numpy as np


def salt_and_pepper(
    data: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Flipea cada pixel con probabilidad p: 0→1 y 1→0.
    Respeta el dominio binario {0, 1}.
    """
    noisy = data.copy()
    mask = rng.random(data.shape) < p
    noisy[mask] = 1.0 - noisy[mask]
    return noisy


def masking(
    data: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Fuerza cada pixel a 0 con probabilidad p.
    Simula oclusión parcial — solo borra, no invierte.
    """
    noisy = data.copy()
    mask = rng.random(data.shape) < p
    noisy[mask] = 0.0
    return noisy
