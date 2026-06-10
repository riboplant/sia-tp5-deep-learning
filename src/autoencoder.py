import numpy as np

from src.layers import Dense, Tanh, Sigmoid
from src.network import Network


class Autoencoder:
    """
    Autoencoder: encoder comprime la entrada al espacio latente,
    decoder la reconstruye.

    El espacio latente (salida del encoder) no tiene activación,
    permitiendo que tome cualquier valor real.
    """

    def __init__(self, encoder: Network, decoder: Network) -> None:
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.decoder.forward(self.encoder.forward(x))

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.encoder.forward(x)

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self.decoder.forward(z)

    def backward(self, grad: np.ndarray) -> None:
        self.encoder.backward(self.decoder.backward(grad))

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.encoder.params() + self.decoder.params()


def build_shallow(seed: int = 42) -> Autoencoder:
    """Arquitectura shallow: 35 -> 16 -> 2 -> 16 -> 35"""
    np.random.seed(seed)
    return Autoencoder(
        encoder=Network([Dense(35, 16), Tanh(), Dense(16, 2)]),
        decoder=Network([Dense(2, 16), Tanh(), Dense(16, 35), Sigmoid()]),
    )


def build_deep(seed: int = 42) -> Autoencoder:
    """Arquitectura deep: 35 -> 32 -> 16 -> 8 -> 2 -> 8 -> 16 -> 32 -> 35"""
    np.random.seed(seed)
    return Autoencoder(
        encoder=Network([
            Dense(35, 32), Tanh(),
            Dense(32, 16), Tanh(),
            Dense(16, 8),  Tanh(),
            Dense(8, 2),
        ]),
        decoder=Network([
            Dense(2, 8),   Tanh(),
            Dense(8, 16),  Tanh(),
            Dense(16, 32), Tanh(),
            Dense(32, 35), Sigmoid(),
        ]),
    )


def build_olivetti(
    input_dim: int,
    latent_dim: int,
    arch: str = "medium",
    seed: int = 42,
) -> Autoencoder:
    """
    Autoencoder para entradas de escala Olivetti (1024–4096 px).

    Arquitecturas:
      shallow: input → 256 → 128 → latent
      medium:  input → 512 → 256 → 128 → latent
      deep:    input → 512 → 256 → 128 → 64 → latent
    """
    np.random.seed(seed)
    _dims = {
        "shallow": [256, 128],
        "medium":  [512, 256, 128],
        "deep":    [512, 256, 128, 64],
    }
    if arch not in _dims:
        raise ValueError(f"arch debe ser uno de {list(_dims)}")
    dims = _dims[arch]

    enc_layers = []
    prev = input_dim
    for d in dims:
        enc_layers += [Dense(prev, d), Tanh()]
        prev = d
    enc_layers.append(Dense(prev, latent_dim))

    dec_layers = []
    prev = latent_dim
    for d in reversed(dims):
        dec_layers += [Dense(prev, d), Tanh()]
        prev = d
    dec_layers += [Dense(prev, input_dim), Sigmoid()]

    return Autoencoder(encoder=Network(enc_layers), decoder=Network(dec_layers))


def build_deep_wide(seed: int = 42) -> Autoencoder:
    """Arquitectura wide: 35 -> 64 -> 32 -> 16 -> 2 -> 16 -> 32 -> 64 -> 35
    Mayor capacidad en capas ocultas para tareas de denoising más agresivas."""
    np.random.seed(seed)
    return Autoencoder(
        encoder=Network([
            Dense(35, 64), Tanh(),
            Dense(64, 32), Tanh(),
            Dense(32, 16), Tanh(),
            Dense(16, 2),
        ]),
        decoder=Network([
            Dense(2, 16),  Tanh(),
            Dense(16, 32), Tanh(),
            Dense(32, 64), Tanh(),
            Dense(64, 35), Sigmoid(),
        ]),
    )
