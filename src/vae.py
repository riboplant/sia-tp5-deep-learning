import numpy as np

from src.layers import Dense, Tanh, Sigmoid
from src.network import Network


class VAEEncoder:
    """
    Encoder probabilístico: x → (μ, log σ²)

    El body transforma la entrada en una representación intermedia h.
    Desde h, dos cabezas lineales independientes producen μ y log σ².
    """

    def __init__(self, body: Network, hidden_dim: int, latent_dim: int) -> None:
        self.body = body
        self.mu_head     = Dense(hidden_dim, latent_dim)
        self.logvar_head = Dense(hidden_dim, latent_dim)
        self._h: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._h = self.body.forward(x)
        mu     = self.mu_head.forward(self._h)
        logvar = self.logvar_head.forward(self._h)
        return mu, logvar

    def backward(self, grad_mu: np.ndarray, grad_logvar: np.ndarray) -> None:
        grad_h = self.mu_head.backward(grad_mu) + self.logvar_head.backward(grad_logvar)
        self.body.backward(grad_h)

    def params(self) -> list:
        return self.body.params() + self.mu_head.params() + self.logvar_head.params()


class VAE:
    """
    Variational Autoencoder.

    Forward (training): encode → reparametrize → decode
    Forward (inference): usa μ directamente como z (sin ruido)
    Backward: gradiente de reconstrucción + gradiente de KL (analítico)
    """

    def __init__(self, encoder: VAEEncoder, decoder: Network) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self._mu:     np.ndarray | None = None
        self._logvar: np.ndarray | None = None
        self._eps:    np.ndarray | None = None
        self._z:      np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        mu, logvar = self.encoder.forward(x)
        self._mu     = mu
        self._logvar = logvar
        if training:
            self._eps = np.random.standard_normal(mu.shape)
            self._z   = mu + np.exp(0.5 * logvar) * self._eps
        else:
            self._z = mu  # en inferencia usamos la media
        return self.decoder.forward(self._z)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Retorna μ como representación latente determinística."""
        mu, _ = self.encoder.forward(x)
        return mu

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self.decoder.forward(z)

    def backward(self, grad_recon: np.ndarray, beta: float = 1.0) -> None:
        """
        Propaga gradientes de reconstrucción + KL.

        grad_recon : dL_recon/dy_hat — viene de la loss de reconstrucción
        beta       : peso del término KL
        """
        # Gradiente desde el decoder
        grad_z = self.decoder.backward(grad_recon)

        # Reparametrización: z = μ + σ·ε  con σ = exp(0.5·logvar)
        # dz/dμ = 1  →  grad_μ_recon = grad_z
        # dz/d(logvar) = 0.5·σ·ε  →  grad_logvar_recon = grad_z · 0.5·σ·ε
        sigma = np.exp(0.5 * self._logvar)
        grad_mu_recon     = grad_z
        grad_logvar_recon = grad_z * 0.5 * sigma * self._eps

        # Gradiente KL: KL = -½·mean(1 + logvar - μ² - exp(logvar))
        # dKL/dμ      =  μ / (B·L)
        # dKL/d(logvar) = 0.5·(exp(logvar) - 1) / (B·L)
        n = float(self._mu.size)  # B × latent_dim
        grad_mu_kl     = beta * self._mu / n
        grad_logvar_kl = beta * 0.5 * (np.exp(self._logvar) - 1.0) / n

        self.encoder.backward(grad_mu_recon + grad_mu_kl,
                               grad_logvar_recon + grad_logvar_kl)

    def params(self) -> list:
        return self.encoder.params() + self.decoder.params()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_vae(input_dim: int = 81, latent_dim: int = 2, seed: int = 42) -> VAE:
    """VAE para caras 9×9: body 81→64→32→16, heads →2, decoder 2→16→32→64→81."""
    np.random.seed(seed)
    hidden_dim = 16
    body = Network([
        Dense(input_dim, 64), Tanh(),
        Dense(64, 32),        Tanh(),
        Dense(32, hidden_dim), Tanh(),
    ])
    encoder = VAEEncoder(body, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Network([
        Dense(latent_dim, hidden_dim), Tanh(),
        Dense(hidden_dim, 32),         Tanh(),
        Dense(32, 64),                 Tanh(),
        Dense(64, input_dim),          Sigmoid(),
    ])
    return VAE(encoder, decoder)
