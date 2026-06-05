from typing import Callable

import numpy as np

from src.autoencoder import Autoencoder


def pixel_error(y_hat: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Píxeles incorrectos por muestra al aplicar umbral."""
    return np.abs((y_hat > threshold).astype(float) - y).sum(axis=1)


def train(
    model: Autoencoder,
    data: np.ndarray,
    optimizer,
    loss_fn,
    epochs: int = 50_000,
    verbose: bool = True,
    log_every: int = 5_000,
) -> dict:
    history: dict = {"loss": [], "max_px_err": [], "converged_at": None}

    for epoch in range(1, epochs + 1):
        y_hat = model.forward(data)
        loss_val = loss_fn(y_hat, data)
        model.backward(loss_fn.grad(y_hat, data))
        optimizer.step(model.params())

        max_err = float(pixel_error(y_hat, data).max())
        history["loss"].append(loss_val)
        history["max_px_err"].append(max_err)

        if verbose and epoch % log_every == 0:
            print(f"  Epoch {epoch:6d} | loss: {loss_val:.5f} | max px err: {max_err:.0f}")

        if max_err <= 1.0 and history["converged_at"] is None:
            history["converged_at"] = epoch
            if verbose:
                print(f"  Convergido en epoch {epoch} | loss: {loss_val:.5f}")
            break

    return history


def train_denoising(
    model: Autoencoder,
    data: np.ndarray,
    optimizer,
    loss_fn,
    noise_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    epochs: int = 50_000,
    seed: int = 42,
    verbose: bool = True,
    log_every: int = 5_000,
) -> dict:
    """
    Entrena un Denoising Autoencoder.

    Cada epoch aplica noise_fn sobre la entrada, pero la loss se calcula
    contra el dato limpio original. El ruido es fresco en cada epoch
    para evitar memorización de un patrón específico.
    """
    rng = np.random.default_rng(seed)
    history: dict = {"loss": [], "max_px_err": [], "converged_at": None}

    for epoch in range(1, epochs + 1):
        noisy = noise_fn(data, rng)          # entrada ruidosa (nueva cada epoch)
        y_hat = model.forward(noisy)          # reconstrucción a partir del ruido
        loss_val = loss_fn(y_hat, data)       # loss contra el original limpio
        model.backward(loss_fn.grad(y_hat, data))
        optimizer.step(model.params())

        max_err = float(pixel_error(y_hat, data).max())
        history["loss"].append(loss_val)
        history["max_px_err"].append(max_err)

        if verbose and epoch % log_every == 0:
            print(f"  Epoch {epoch:6d} | loss: {loss_val:.5f} | max px err: {max_err:.0f}")

        if max_err <= 1.0 and history["converged_at"] is None:
            history["converged_at"] = epoch
            if verbose:
                print(f"  Convergido en epoch {epoch} | loss: {loss_val:.5f}")
            break

    return history
