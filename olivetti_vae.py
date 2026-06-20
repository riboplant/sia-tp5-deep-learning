"""
VAE sobre Olivetti Faces (32×32) — análisis completo.

Experimentos:
  1. Curvas de entrenamiento (loss / recon / KL)
  2. Reconstrucciones (original vs reconstruido vs diferencia)
  3. Espacio latente — PCA 2D coloreado por persona
  4. Generación — 8 caras muestreadas del prior N(0, I_8)
  5. Interpolación entre dos personas en el espacio latente
  6. AE vs VAE — estructura latente y calidad generativa
"""
import numpy as np

from src.olivetti import load_olivetti
from src.vae import build_olivetti_vae
from src.autoencoder import build_olivetti
from src.losses import VAELoss, MSE
from src.optimizers import Adam
from src.training import train_vae_continuous, train_continuous
from src.visualization import (
    set_results_dir,
    plot_vae_training_continuous,
    plot_face_reconstructions,
    plot_latent_pca,
    plot_vae_samples,
    plot_face_interpolation,
    plot_ae_vs_vae_olivetti,
)

SEED     = 42
SIZE     = 32
LATENT   = 8
BETA     = 0.008   # β_natural ≈ L/F = 8/1024
WARMUP   = 2_000
EPOCHS   = 10_000
BATCH    = 50
LR       = 1e-3
PATIENCE = 500

set_results_dir("results/olivetti")


def main() -> None:
    data, labels = load_olivetti(size=SIZE)
    input_dim = data.shape[1]
    print(f"Dataset: {data.shape[0]} caras  {SIZE}×{SIZE}  ({input_dim}px)")

    # ------------------------------------------------------------------ #
    # Entrenar VAE                                                         #
    # ------------------------------------------------------------------ #
    print(f"\nEntrenando VAE (latente={LATENT}D, β={BETA}, warmup={WARMUP})...")
    vae = build_olivetti_vae(input_dim=input_dim, latent_dim=LATENT, seed=SEED)
    history = train_vae_continuous(
        vae, data, Adam(lr=LR), VAELoss(beta=BETA, recon="mse"),
        epochs=EPOCHS, batch_size=BATCH, beta_warmup=WARMUP,
        seed=SEED, patience=PATIENCE, log_every=1_000,
    )
    conv = history["converged_at"]
    print(f"\n  {'Early stop en época ' + str(conv) if conv else 'max_epochs alcanzado'}")
    print(f"  Loss: {history['loss'][-1]:.5f}  "
          f"Recon: {history['recon'][-1]:.5f}  "
          f"KL: {history['kl'][-1]:.4f}")

    # ------------------------------------------------------------------ #
    # 1. Curvas de entrenamiento                                           #
    # ------------------------------------------------------------------ #
    print("\nExp 1 — curvas de entrenamiento...")
    plot_vae_training_continuous(history, filename="vae_training.png")

    # ------------------------------------------------------------------ #
    # 2. Reconstrucciones                                                  #
    # ------------------------------------------------------------------ #
    print("Exp 2 — reconstrucciones...")
    recon = vae.forward(data, training=False)
    plot_face_reconstructions(data, recon, SIZE, n_show=8,
                              title="VAE Olivetti — reconstrucciones",
                              filename="vae_reconstructions.png")

    # ------------------------------------------------------------------ #
    # 3. Espacio latente — PCA                                             #
    # ------------------------------------------------------------------ #
    print("Exp 3 — espacio latente PCA...")
    latent_vae = vae.encode(data)
    plot_latent_pca(latent_vae, labels,
                    title="VAE Olivetti — espacio latente 8D (PCA)",
                    filename="vae_latent_pca.png")

    # ------------------------------------------------------------------ #
    # 4. Generación desde el prior N(0, I_8)                              #
    # ------------------------------------------------------------------ #
    print("Exp 4 — generación desde prior...")
    plot_vae_samples(vae, rows=1, cols=8,
                     shape=(SIZE, SIZE), threshold=None,
                     filename="vae_samples_prior.png")

    # ------------------------------------------------------------------ #
    # 5. Interpolación entre dos personas                                  #
    # ------------------------------------------------------------------ #
    print("Exp 5 — interpolación entre personas...")
    # Tomar la primera foto de persona 0 y persona 5 (visualmente distintas)
    idx_a = int(np.where(labels == 0)[0][0])
    idx_b = int(np.where(labels == 5)[0][0])
    plot_face_interpolation(
        vae, latent_vae,
        idx_a, idx_b,
        label_a=f"Persona {labels[idx_a]}",
        label_b=f"Persona {labels[idx_b]}",
        n_steps=9, shape=(SIZE, SIZE), threshold=None,
        filename="vae_interpolation.png",
    )

    # ------------------------------------------------------------------ #
    # 6. AE vs VAE                                                         #
    # ------------------------------------------------------------------ #
    print("Exp 6 — entrenando AE para comparación...")
    ae = build_olivetti(input_dim, latent_dim=LATENT, arch="deep", seed=SEED)
    train_continuous(
        ae, data, Adam(lr=LR), MSE(),
        epochs=EPOCHS, batch_size=BATCH, seed=SEED,
        patience=PATIENCE, verbose=True, log_every=2_000,
    )
    latent_ae = ae.encode(data)
    plot_ae_vs_vae_olivetti(
        ae, vae, data, labels,
        latent_ae, latent_vae,
        size=SIZE, n_samples=8,
        filename="vae_vs_ae.png",
    )

    print("\nListo. Todas las figuras en results/olivetti/")


if __name__ == "__main__":
    main()
