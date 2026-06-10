"""
Estudio ceteris paribus del autoencoder sobre Olivetti Faces (32×32).

Experimento 1 — Dimensión latente  : {2, 8, 16, 32}
Experimento 2 — Arquitectura        : {shallow, medium, deep}
Experimento 3 — Loss function       : {MSE, BCE}

Al final del estudio se reporta la configuración óptima.
"""
import numpy as np

from src.olivetti import load_olivetti
from src.autoencoder import build_olivetti
from src.losses import MSE, BCE
from src.optimizers import Adam
from src.training import train_continuous
from src.visualization import (
    set_results_dir,
    plot_olivetti_dataset,
    plot_face_reconstructions,
    plot_olivetti_training,
    plot_latent_faces_2d,
    plot_latent_pca,
)

SEED       = 42
SIZE       = 32
EPOCHS     = 5_000
BATCH      = 50
LR         = 1e-3
PATIENCE   = 600

set_results_dir("results/olivetti")


def mse_eval(model, data: np.ndarray) -> float:
    """MSE de reconstrucción como métrica común (independiente de la loss usada)."""
    recon = model.forward(data)
    return float(np.mean((recon - data) ** 2))


# --------------------------------------------------------------------------- #
# Exp 1: Dimensión latente                                                      #
# --------------------------------------------------------------------------- #

def run_exp1(data: np.ndarray, labels: np.ndarray, input_dim: int) -> int:
    print("\n" + "=" * 60)
    print("Exp 1: Dimensión latente  (arch=medium, loss=MSE)")
    print("=" * 60)

    latent_dims = [2, 8, 16, 32]
    results = {}

    for ld in latent_dims:
        print(f"  latente={ld}D...", end="  ", flush=True)
        model = build_olivetti(input_dim, ld, arch="medium", seed=SEED)
        history = train_continuous(
            model, data, Adam(lr=LR), MSE(),
            epochs=EPOCHS, batch_size=BATCH, seed=SEED,
            verbose=True, log_every=EPOCHS + 1,  # sin logs intermedios
            patience=PATIENCE,
        )
        mse = mse_eval(model, data)
        print(f"MSE={mse:.5f}  (loss final={history['loss'][-1]:.5f})")
        results[f"latent={ld}"] = {"model": model, "history": history,
                                    "latent_dim": ld, "mse": mse}

    plot_olivetti_training(results, "Exp 1: Dimensión latente (arch=medium, loss=MSE)",
                           filename="exp1_latent_dim.png")

    # Reconstrucciones para cada config
    for key, r in results.items():
        recon = r["model"].forward(data)
        plot_face_reconstructions(data, recon, SIZE, n_show=8,
                                  title=key, filename=f"exp1_recon_{key.replace('=', '')}.png")

    # Espacio latente: 2D con thumbnails, ND con PCA
    for key, r in results.items():
        ld = r["latent_dim"]
        latent_enc = r["model"].encode(data)
        if ld == 2:
            plot_latent_faces_2d(latent_enc, data, labels, SIZE,
                                  filename="exp1_latent_2d.png")
        else:
            plot_latent_pca(latent_enc, labels,
                            title=f"Latente {ld}D — PCA",
                            filename=f"exp1_latent_pca_{ld}d.png")

    best_key = min(results, key=lambda k: results[k]["mse"])
    best_ld  = results[best_key]["latent_dim"]
    print(f"\nMejor dimensión latente: {best_ld}D  (MSE={results[best_key]['mse']:.5f})")
    return best_ld


# --------------------------------------------------------------------------- #
# Exp 2: Arquitectura                                                           #
# --------------------------------------------------------------------------- #

def run_exp2(data: np.ndarray, labels: np.ndarray, input_dim: int, best_latent: int) -> str:
    print("\n" + "=" * 60)
    print(f"Exp 2: Arquitectura  (latente={best_latent}D, loss=MSE)")
    print("=" * 60)

    archs = ["shallow", "medium", "deep"]
    results = {}

    for arch in archs:
        print(f"  {arch}...", end="  ", flush=True)
        model = build_olivetti(input_dim, best_latent, arch=arch, seed=SEED)
        history = train_continuous(
            model, data, Adam(lr=LR), MSE(),
            epochs=EPOCHS, batch_size=BATCH, seed=SEED,
            verbose=True, log_every=EPOCHS + 1,
            patience=PATIENCE,
        )
        mse = mse_eval(model, data)
        print(f"MSE={mse:.5f}  (loss final={history['loss'][-1]:.5f})")
        results[arch] = {"model": model, "history": history, "mse": mse}

    plot_olivetti_training(results,
                           f"Exp 2: Arquitectura (latente={best_latent}D, loss=MSE)",
                           filename="exp2_arquitectura.png")

    for key, r in results.items():
        recon = r["model"].forward(data)
        plot_face_reconstructions(data, recon, SIZE, n_show=8,
                                  title=f"arch={key}, latente={best_latent}D",
                                  filename=f"exp2_recon_{key}.png")

    best_arch = min(results, key=lambda k: results[k]["mse"])
    print(f"\nMejor arquitectura: {best_arch}  (MSE={results[best_arch]['mse']:.5f})")
    return best_arch


# --------------------------------------------------------------------------- #
# Exp 3: Loss function                                                          #
# --------------------------------------------------------------------------- #

def run_exp3(data: np.ndarray, input_dim: int, best_latent: int, best_arch: str) -> str:
    print("\n" + "=" * 60)
    print(f"Exp 3: Loss function  (latente={best_latent}D, arch={best_arch})")
    print("=" * 60)

    loss_fns = {"MSE": MSE(), "BCE": BCE()}
    results = {}

    for loss_name, loss_fn in loss_fns.items():
        print(f"  {loss_name}...", end="  ", flush=True)
        model = build_olivetti(input_dim, best_latent, arch=best_arch, seed=SEED)
        history = train_continuous(
            model, data, Adam(lr=LR), loss_fn,
            epochs=EPOCHS, batch_size=BATCH, seed=SEED,
            verbose=True, log_every=EPOCHS + 1,
            patience=PATIENCE,
        )
        mse = mse_eval(model, data)
        print(f"MSE={mse:.5f}  (loss propia final={history['loss'][-1]:.5f})")
        results[loss_name] = {"model": model, "history": history, "mse": mse}

    plot_olivetti_training(results,
                           f"Exp 3: Loss function (latente={best_latent}D, arch={best_arch})",
                           filename="exp3_loss_fn.png")

    for key, r in results.items():
        recon = r["model"].forward(data)
        plot_face_reconstructions(data, recon, SIZE, n_show=8,
                                  title=f"loss={key}",
                                  filename=f"exp3_recon_{key.lower()}.png")

    print(f"\n  MSE comparación (métrica común):")
    for name, r in results.items():
        print(f"    {name}: {r['mse']:.5f}")

    best_loss = min(results, key=lambda k: results[k]["mse"])
    print(f"\nMejor loss: {best_loss}  (MSE={results[best_loss]['mse']:.5f})")
    return best_loss


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    data, labels = load_olivetti(size=SIZE)
    input_dim = data.shape[1]
    print(f"Dataset: {data.shape[0]} caras de {input_dim} píxeles ({SIZE}×{SIZE})")
    print(f"Personas: 40  |  Fotos por persona: 10\n")

    # Visualizar dataset
    plot_olivetti_dataset(data, labels, SIZE, filename="dataset.png")

    best_latent = run_exp1(data, labels, input_dim)
    best_arch   = run_exp2(data, labels, input_dim, best_latent)
    best_loss   = run_exp3(data, input_dim, best_latent, best_arch)

    print("\n" + "=" * 60)
    print("Configuración óptima encontrada:")
    print(f"  Dimensión latente : {best_latent}D")
    print(f"  Arquitectura      : {best_arch}")
    print(f"  Loss function     : {best_loss}")
    print(f"\nFiguras en results/olivetti/")
    print("=" * 60)


if __name__ == "__main__":
    main()
