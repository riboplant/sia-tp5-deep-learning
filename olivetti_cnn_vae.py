"""
VAE convolucional (CNN) sobre Olivetti Faces (32×32).

Encoder: Conv(1→8)→Pool → Conv(8→16)→Pool → Conv(16→32)→Pool → Flatten → Dense(128) → μ/logσ
Decoder: Dense → Dense → Reshape(32,4,4) → [Upsample+Conv]×3 → Sigmoid → Flatten

Experimentos 1-5: análogos al Dense-VAE para comparación directa de figuras.
Exp 6: comparación de reconstrucciones Dense-VAE vs CNN-VAE sobre las mismas imágenes.
"""
import numpy as np

from src.olivetti import load_olivetti
from src.vae import build_cnn_vae, build_olivetti_vae
from src.losses import VAELoss
from src.optimizers import Adam
from src.training import train_vae_continuous
from src.visualization import (
    set_results_dir,
    plot_vae_training_continuous,
    plot_face_reconstructions,
    plot_latent_pca,
    plot_vae_samples,
    plot_face_interpolation,
)

SEED     = 42
SIZE     = 32
LATENT   = 32
BETA     = 0.03125   # β_natural = 32/1024
WARMUP   = 2_000
EPOCHS   = 10_000
BATCH    = 50
LR       = 1e-3
PATIENCE = 300       # más agresivo que el Dense-VAE (500)

set_results_dir("results/olivetti")


def main() -> None:
    data, labels = load_olivetti(size=SIZE)
    input_dim = data.shape[1]
    print(f"Dataset: {data.shape[0]} caras  {SIZE}×{SIZE}  ({input_dim}px)")

    # ------------------------------------------------------------------ #
    # Entrenar CNN-VAE                                                     #
    # ------------------------------------------------------------------ #
    print(f"\nEntrenando CNN-VAE (latente={LATENT}D, β={BETA}, warmup={WARMUP})...")
    cnn_vae = build_cnn_vae(size=SIZE, latent_dim=LATENT, seed=SEED)
    history = train_vae_continuous(
        cnn_vae, data, Adam(lr=LR), VAELoss(beta=BETA, recon="mse"),
        epochs=EPOCHS, batch_size=BATCH, beta_warmup=WARMUP,
        seed=SEED, patience=PATIENCE, log_every=1_000,
    )
    conv = history["converged_at"]
    print(f"\n  {'Early stop en época ' + str(conv) if conv else 'max_epochs alcanzado'}")
    print(f"  Loss: {history['loss'][-1]:.5f}  "
          f"Recon: {history['recon'][-1]:.5f}  "
          f"KL: {history['kl'][-1]:.4f}")

    recon_cnn = cnn_vae.forward(data, training=False)
    mse_cnn = float(np.mean((recon_cnn - data) ** 2))
    print(f"  MSE eval: {mse_cnn:.5f}")

    # ------------------------------------------------------------------ #
    # 1. Curvas de entrenamiento                                           #
    # ------------------------------------------------------------------ #
    print("\nExp 1 — curvas de entrenamiento...")
    plot_vae_training_continuous(history, filename="cnn_vae_training.png")

    # ------------------------------------------------------------------ #
    # 2. Reconstrucciones                                                  #
    # ------------------------------------------------------------------ #
    print("Exp 2 — reconstrucciones...")
    plot_face_reconstructions(data, recon_cnn, SIZE, n_show=8,
                              title="CNN-VAE Olivetti — reconstrucciones",
                              filename="cnn_vae_reconstructions.png")

    # ------------------------------------------------------------------ #
    # 3. Espacio latente — PCA                                             #
    # ------------------------------------------------------------------ #
    print("Exp 3 — espacio latente PCA...")
    latent_cnn = cnn_vae.encode(data)
    plot_latent_pca(latent_cnn, labels,
                    title="CNN-VAE Olivetti — espacio latente 32D (PCA)",
                    filename="cnn_vae_latent_pca.png")

    # ------------------------------------------------------------------ #
    # 4. Generación desde el prior                                         #
    # ------------------------------------------------------------------ #
    print("Exp 4 — generación desde prior...")
    plot_vae_samples(cnn_vae, rows=1, cols=8,
                     shape=(SIZE, SIZE), threshold=None,
                     filename="cnn_vae_samples_prior.png")

    # ------------------------------------------------------------------ #
    # 5. Interpolación entre dos personas                                  #
    # ------------------------------------------------------------------ #
    print("Exp 5 — interpolación entre personas...")
    idx_a = int(np.where(labels == 0)[0][0])
    idx_b = int(np.where(labels == 5)[0][0])
    plot_face_interpolation(
        cnn_vae, latent_cnn,
        idx_a, idx_b,
        label_a=f"Persona {labels[idx_a]}",
        label_b=f"Persona {labels[idx_b]}",
        n_steps=9, shape=(SIZE, SIZE), threshold=None,
        filename="cnn_vae_interpolation.png",
    )

    # ------------------------------------------------------------------ #
    # 6. Comparación Dense-VAE vs CNN-VAE (reconstrucciones lado a lado)  #
    # ------------------------------------------------------------------ #
    print("Exp 6 — entrenando Dense-VAE para comparación...")
    dense_vae = build_olivetti_vae(input_dim=input_dim, latent_dim=LATENT, seed=SEED)
    dense_history = train_vae_continuous(
        dense_vae, data, Adam(lr=LR), VAELoss(beta=BETA, recon="mse"),
        epochs=EPOCHS, batch_size=BATCH, beta_warmup=WARMUP,
        seed=SEED, patience=PATIENCE, verbose=True, log_every=2_000,
    )
    recon_dense = dense_vae.forward(data, training=False)
    mse_dense = float(np.mean((recon_dense - data) ** 2))

    import matplotlib.pyplot as plt
    n_show = 8
    indices = np.linspace(0, len(data) - 1, n_show, dtype=int)
    shape = (SIZE, SIZE)
    fig, axes = plt.subplots(4, n_show, figsize=(n_show * 1.6, 7))
    row_labels = ["Original", "Dense-VAE", "CNN-VAE", "|CNN - Dense|"]
    for col, idx in enumerate(indices):
        orig  = data[idx].reshape(shape)
        r_d   = recon_dense[idx].reshape(shape)
        r_c   = recon_cnn[idx].reshape(shape)
        imgs  = [orig, r_d, r_c, np.abs(r_c - r_d)]
        cmaps = ["gray", "gray", "gray", "hot"]
        for row, (img, cmap) in enumerate(zip(imgs, cmaps)):
            axes[row, col].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[row, col].axis("off")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=60, va="center")
    fig.suptitle(
        f"Dense-VAE (MSE={mse_dense:.5f}) vs CNN-VAE (MSE={mse_cnn:.5f})", fontsize=12
    )
    plt.tight_layout()
    plt.savefig("results/olivetti/cnn_vs_dense_vae.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [guardado] results/olivetti/cnn_vs_dense_vae.png")

    print("\n" + "=" * 50)
    print(f"Dense-VAE MSE : {mse_dense:.5f}")
    print(f"CNN-VAE   MSE : {mse_cnn:.5f}")
    mejora = (mse_dense - mse_cnn) / mse_dense * 100
    if mejora > 0:
        print(f"Mejora CNN    : +{mejora:.1f}%")
    else:
        print(f"Dense sigue mejor en {-mejora:.1f}%")
    print("\nListo. Figuras en results/olivetti/")


if __name__ == "__main__":
    main()
