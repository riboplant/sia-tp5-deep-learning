from typing import Callable

import numpy as np

from src.autoencoder import Autoencoder
from src.losses import VAELoss


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


def train_vae(
    model,
    data: np.ndarray,
    optimizer,
    loss_fn: VAELoss,
    epochs: int = 100_000,
    beta_warmup: int = 5_000,
    seed: int = 42,
    verbose: bool = True,
    log_every: int = 10_000,
) -> dict:
    """
    Entrena un VAE con KL annealing.

    Durante las primeras beta_warmup épocas β sube linealmente de 0 a loss_fn.beta.
    Esto permite que el modelo aprenda primero a reconstruir antes de regularizar.

    history incluye 'recon', 'kl' y 'loss' por separado para análisis.
    """
    np.random.seed(seed)
    history: dict = {
        "loss": [], "recon": [], "kl": [],
        "max_px_err": [], "converged_at": None,
    }
    beta_max = loss_fn.beta

    for epoch in range(1, epochs + 1):
        beta = beta_max * min(1.0, epoch / beta_warmup)

        y_hat = model.forward(data, training=True)
        loss_val = loss_fn(y_hat, data, model._mu, model._logvar)
        model.backward(loss_fn.grad(y_hat, data), beta=beta)
        optimizer.step(model.params())

        max_err = float(pixel_error(y_hat, data).max())
        history["loss"].append(loss_val)
        history["recon"].append(loss_fn.last_recon)
        history["kl"].append(loss_fn.last_kl)
        history["max_px_err"].append(max_err)

        if verbose and epoch % log_every == 0:
            print(
                f"  Epoch {epoch:7d} | β={beta:.3f} | "
                f"loss: {loss_val:.4f} (recon: {loss_fn.last_recon:.4f} "
                f"kl: {loss_fn.last_kl:.4f}) | max px err: {max_err:.0f}"
            )

        if max_err <= 1.0 and history["converged_at"] is None:
            history["converged_at"] = epoch
            if verbose:
                print(f"  Convergido en epoch {epoch} | loss: {loss_val:.5f}")
            break

    return history


def train_continuous(
    model: Autoencoder,
    data: np.ndarray,
    optimizer,
    loss_fn,
    epochs: int = 5_000,
    batch_size: int = 50,
    seed: int = 42,
    verbose: bool = True,
    log_every: int = 500,
    patience: int = 600,
    tol: float = 1e-6,
) -> dict:
    """
    Entrena un autoencoder sobre imágenes continuas con mini-batches.
    Early stopping por plateau de loss en lugar de error de píxel.
    """
    rng = np.random.default_rng(seed)
    history: dict = {"loss": [], "converged_at": None}
    n = len(data)
    best_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = data[indices[start : start + batch_size]]
            y_hat = model.forward(batch)
            loss_val = loss_fn(y_hat, batch)
            model.backward(loss_fn.grad(y_hat, batch))
            optimizer.step(model.params())
            epoch_loss += loss_val
            n_batches += 1
        epoch_loss /= n_batches
        history["loss"].append(epoch_loss)

        if verbose and epoch % log_every == 0:
            print(f"  Epoch {epoch:6d} | loss: {epoch_loss:.6f}")

        if best_loss - epoch_loss > tol:
            best_loss = epoch_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                history["converged_at"] = epoch
                if verbose:
                    print(f"  Early stop epoch {epoch} | loss: {epoch_loss:.6f}")
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
