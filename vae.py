import numpy as np

from src.faces import load_faces, MOUTH_VALUES, EYE_VALUES, H, W
from src.vae import build_vae
from src.autoencoder import build_deep
from src.losses import VAELoss, BCE
from src.optimizers import Adam
from src.training import train_vae, train, pixel_error
from src.visualization import (
    plot_chars,
    plot_latent_space,
    plot_vae_training,
    plot_latent_by_attribute,
    plot_vae_samples,
    plot_face_interpolation,
    plot_ae_vs_vae,
)

SEED       = 42
MAX_EPOCHS = 100_000
BETA       = 0.01   # BCE / (B*F) = 1/4374,  KL / (B*L) = 1/108 -> β ≈ L/F ≈ 0.025
WARMUP     = 5_000
FACE_SHAPE = (H, W)  # (9, 9)


def main() -> None:
    data, labels, attrs = load_faces()
    print(f"Dataset: {data.shape[0]} caras de {data.shape[1]} píxeles ({H}×{W})")

    # ------------------------------------------------------------------ #
    # Visualizar el dataset                                                #
    # ------------------------------------------------------------------ #
    plot_chars(list(data), labels, cols=9, shape=FACE_SHAPE,
               title="Dataset: 54 caras pixeladas (9×9)", filename="vae_dataset.png")

    # ------------------------------------------------------------------ #
    # Entrenar VAE                                                         #
    # ------------------------------------------------------------------ #
    print(f"\nEntrenando VAE (β={BETA}, warmup={WARMUP}, max_epochs={MAX_EPOCHS})...")
    vae = build_vae(input_dim=data.shape[1], latent_dim=2, seed=SEED)
    optimizer = Adam(lr=1e-3)
    loss_fn = VAELoss(beta=BETA)

    history = train_vae(
        vae, data, optimizer, loss_fn,
        epochs=MAX_EPOCHS, beta_warmup=WARMUP, seed=SEED,
    )

    conv = history["converged_at"]
    print(f"\nResultado VAE: {'convergió en época ' + str(conv) if conv else 'no convergió'}")
    print(f"  Loss final  : {history['loss'][-1]:.5f}")
    print(f"  Recon final : {history['recon'][-1]:.5f}")
    print(f"  KL final    : {history['kl'][-1]:.5f}")
    print(f"  Max px err  : {history['max_px_err'][-1]:.0f}")

    plot_vae_training(history, filename="vae_training.png")

    # ------------------------------------------------------------------ #
    # Reconstrucciones                                                     #
    # ------------------------------------------------------------------ #
    reconstructed = vae.forward(data, training=False)
    errors = pixel_error(reconstructed, data)
    print(f"\nError de reconstrucción — max: {errors.max():.0f} px | mean: {errors.mean():.2f} px")

    plot_chars(list(reconstructed), labels, cols=9, shape=FACE_SHAPE,
               threshold=0.5, title="Reconstrucciones VAE", filename="vae_reconstructions.png")

    # ------------------------------------------------------------------ #
    # Espacio latente                                                      #
    # ------------------------------------------------------------------ #
    latent_vae = vae.encode(data)  # usa μ
    plot_latent_space(latent_vae, labels, filename="vae_latent.png")
    plot_latent_by_attribute(latent_vae, attrs, filename="vae_latent_attrs.png")

    # ------------------------------------------------------------------ #
    # Generación: muestreo del prior N(0,I)                               #
    # ------------------------------------------------------------------ #
    print("\nGenerando caras nuevas desde el prior N(0,I)...")
    plot_vae_samples(vae, rows=1, cols=8, shape=FACE_SHAPE, filename="vae_samples_prior.png")

    # ------------------------------------------------------------------ #
    # Interpolaciones entre pares de caras                                 #
    # ------------------------------------------------------------------ #
    def find(mouth=None, eyes=None, eyebrows=None, accessory=None):
        for i, a in enumerate(attrs):
            if (mouth is None or a["mouth"] == mouth) and \
               (eyes is None or a["eyes"] == eyes) and \
               (eyebrows is None or a["eyebrows"] == eyebrows) and \
               (accessory is None or a["accessory"] == accessory):
                return i
        return 0

    pairs = [
        (find(mouth="smile",   eyes="open",   eyebrows="raised"),
         find(mouth="frown",   eyes="closed", eyebrows="angry"),
         "feliz", "triste"),
        (find(mouth="neutral", eyes="open",   eyebrows="neutral", accessory="none"),
         find(mouth="neutral", eyes="open",   eyebrows="neutral", accessory="glasses"),
         "sin anteojos", "con anteojos"),
        (find(mouth="smile",   eyes="open",   eyebrows="neutral"),
         find(mouth="smile",   eyes="closed", eyebrows="neutral"),
         "ojos abiertos", "ojos cerrados"),
    ]

    for idx_a, idx_b, la, lb in pairs:
        fname = f"vae_interp_{la.replace(' ','_')}_{lb.replace(' ','_')}.png"
        plot_face_interpolation(vae, latent_vae, idx_a, idx_b, la, lb,
                                shape=FACE_SHAPE, filename=fname)

    # ------------------------------------------------------------------ #
    # Comparación AE vs VAE                                                #
    # ------------------------------------------------------------------ #
    print("\nEntrenando AE básico para comparación...")
    from src.autoencoder import Autoencoder
    from src.network import Network
    from src.layers import Dense, Tanh, Sigmoid

    np.random.seed(SEED)
    ae = Autoencoder(
        encoder=Network([
            Dense(data.shape[1], 64), Tanh(),
            Dense(64, 32), Tanh(),
            Dense(32, 16), Tanh(),
            Dense(16, 2),
        ]),
        decoder=Network([
            Dense(2, 16), Tanh(),
            Dense(16, 32), Tanh(),
            Dense(32, 64), Tanh(),
            Dense(64, data.shape[1]), Sigmoid(),
        ]),
    )
    ae_history = train(ae, data, Adam(lr=1e-3), BCE(),
                       epochs=MAX_EPOCHS, log_every=20_000)

    latent_ae = ae.encode(data)
    plot_ae_vs_vae(ae, vae, latent_ae, latent_vae,
                   n_samples=16, shape=FACE_SHAPE, filename="vae_vs_ae.png")

    print("\nFinalizado. Todas las figuras guardadas en results/")


if __name__ == "__main__":
    main()
