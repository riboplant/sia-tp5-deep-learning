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
