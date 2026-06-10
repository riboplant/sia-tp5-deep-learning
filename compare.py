"""
Experimentos comparativos: AE básico vs VAE sobre el dataset de caras.

Tres experimentos:
  A) Estructura del espacio latente: puntos fijos (AE) vs distribuciones μ±σ (VAE)
  B) Grid decode: el AE produce artefactos en zonas sin datos, el VAE no
  C) Interpolación: el AE puede cruzar zonas muertas, el VAE es siempre suave
"""
import numpy as np

from src.faces import load_faces, H, W
from src.vae import build_vae
from src.autoencoder import Autoencoder
from src.network import Network
from src.layers import Dense, Tanh, Sigmoid
from src.losses import VAELoss, BCE
from src.optimizers import Adam
from src.training import train_vae, train
from src.visualization import (
    plot_ae_vs_vae_latent,
    plot_ae_vs_vae_grid,
    plot_ae_vs_vae_interpolation,
)

SEED       = 42
MAX_EPOCHS = 100_000
FACE_SHAPE = (H, W)


def build_ae(input_dim: int = 81, seed: int = 42) -> Autoencoder:
    np.random.seed(seed)
    return Autoencoder(
        encoder=Network([
            Dense(input_dim, 64), Tanh(),
            Dense(64, 32),        Tanh(),
            Dense(32, 16),        Tanh(),
            Dense(16, 2),
        ]),
        decoder=Network([
            Dense(2, 16),         Tanh(),
            Dense(16, 32),        Tanh(),
            Dense(32, 64),        Tanh(),
            Dense(64, input_dim), Sigmoid(),
        ]),
    )


def main() -> None:
    data, labels, attrs = load_faces()
    print(f"Dataset: {data.shape[0]} caras de {data.shape[1]} píxeles\n")

    # ------------------------------------------------------------------ #
    # Entrenar AE básico                                                   #
    # ------------------------------------------------------------------ #
    print("Entrenando AE básico...")
    ae = build_ae(data.shape[1], seed=SEED)
    np.random.seed(SEED)
    train(ae, data, Adam(lr=1e-3), BCE(), epochs=MAX_EPOCHS, log_every=20_000)

    # ------------------------------------------------------------------ #
    # Entrenar VAE                                                         #
    # ------------------------------------------------------------------ #
    print("\nEntrenando VAE...")
    vae = build_vae(input_dim=data.shape[1], latent_dim=2, seed=SEED)
    train_vae(vae, data, Adam(lr=1e-3), VAELoss(beta=0.01),
              epochs=MAX_EPOCHS, beta_warmup=5_000, seed=SEED, log_every=20_000)

    # Espacios latentes
    latent_ae  = ae.encode(data)
    latent_vae = vae.encode(data)       # usa μ
    _, logvar_vae = vae.encoder.forward(data)

    # ------------------------------------------------------------------ #
    # Experimento A: Estructura del espacio latente                        #
    # ------------------------------------------------------------------ #
    print("\nExperimento A: estructura del espacio latente...")
    plot_ae_vs_vae_latent(latent_ae, latent_vae, logvar_vae, attrs,
                          filename="cmp_latent_structure.png")

    # ------------------------------------------------------------------ #
    # Experimento B: Grid decode                                           #
    # ------------------------------------------------------------------ #
    print("Experimento B: grid decode...")
    plot_ae_vs_vae_grid(ae, vae, latent_ae, latent_vae, FACE_SHAPE,
                        n=10, filename="cmp_grid_decode.png")

    # ------------------------------------------------------------------ #
    # Experimento C: Interpolación con zona muerta                         #
    # ------------------------------------------------------------------ #
    print("Experimento C: interpolación...")
    plot_ae_vs_vae_interpolation(ae, vae, latent_ae, latent_vae, attrs,
                                 FACE_SHAPE, filename="cmp_interpolation.png")

    print("\nListo. Figuras guardadas en results/")


if __name__ == "__main__":
    main()
