"""
Estudio ceteris paribus — Dimensión latente del VAE sobre Olivetti Faces (32×32).

Variar: latent_dim ∈ {8, 16, 32, 64}
Fijo:   arquitectura deep, loss MSE, Adam lr=1e-3
β ajustado por config: β = latent_dim / input_dim  (escala natural)
"""
import numpy as np

from src.olivetti import load_olivetti
from src.vae import build_olivetti_vae
from src.losses import VAELoss
from src.optimizers import Adam
from src.training import train_vae_continuous
from src.visualization import (
    set_results_dir,
    plot_face_reconstructions,
    plot_vae_latent_study,
)

SEED     = 42
SIZE     = 32
EPOCHS   = 10_000
BATCH    = 50
LR       = 1e-3
WARMUP   = 2_000
PATIENCE = 500

LATENT_DIMS = [8, 16, 32, 64]

set_results_dir("results/olivetti")


def mse_eval(model, data: np.ndarray) -> float:
    recon = model.forward(data, training=False)
    return float(np.mean((recon - data) ** 2))


def main() -> None:
    data, labels = load_olivetti(size=SIZE)
    input_dim = data.shape[1]
    print(f"Dataset: {data.shape[0]} caras  {SIZE}×{SIZE}  ({input_dim}px)")
    print(f"\nEstudio ceteris paribus — latent_dim ∈ {LATENT_DIMS}")
    print("=" * 60)

    results = {}

    for ld in LATENT_DIMS:
        beta = ld / input_dim  # β_natural = latent_dim / input_dim
        name = f"latente={ld}D"
        print(f"\n{name}  (β={beta:.4f})...")

        model = build_olivetti_vae(input_dim=input_dim, latent_dim=ld, seed=SEED)
        history = train_vae_continuous(
            model, data, Adam(lr=LR), VAELoss(beta=beta, recon="mse"),
            epochs=EPOCHS, batch_size=BATCH, beta_warmup=WARMUP,
            seed=SEED, patience=PATIENCE, log_every=2_000,
        )

        mse = mse_eval(model, data)
        conv = history["converged_at"] or len(history["loss"])
        print(f"  MSE={mse:.5f}  KL={history['kl'][-1]:.3f}  épocas={conv}")

        results[name] = {"model": model, "history": history, "mse": mse}

        # Reconstrucciones individuales por config
        recon = model.forward(data, training=False)
        plot_face_reconstructions(
            data, recon, SIZE, n_show=8,
            title=f"VAE {name} — reconstrucciones",
            filename=f"vae_study_recon_latent{ld}.png",
        )

    # Figura resumen comparativa
    print("\nGenerando figura resumen...")
    plot_vae_latent_study(
        results, data, labels, size=SIZE, n_samples=8,
        filename="vae_study_latent_dim.png",
    )

    # Tabla resumen
    print("\n" + "=" * 60)
    print("Configuración óptima:")
    print(f"{'Config':<18} {'MSE':>10} {'KL final':>10} {'Épocas':>8}")
    print("-" * 50)
    for name, r in results.items():
        h = r["history"]
        epochs = h["converged_at"] or len(h["loss"])
        print(f"{name:<18} {r['mse']:>10.5f} {h['kl'][-1]:>10.3f} {epochs:>8}")

    best = min(results, key=lambda k: results[k]["mse"])
    print(f"\nMejor MSE: {best}  ({results[best]['mse']:.5f})")
    print("\nFiguras en results/olivetti/")


if __name__ == "__main__":
    main()
