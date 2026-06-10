import numpy as np


class MSE:
    """Mean Squared Error: L = mean((y_hat - y)^2)"""

    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((y_hat - y) ** 2))

    def grad(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2.0 * (y_hat - y) / y_hat.size


class BCE:
    """Binary Cross Entropy: L = -mean(y*log(y_hat) + (1-y)*log(1-y_hat))"""

    _eps = 1e-7

    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        yh = np.clip(y_hat, self._eps, 1.0 - self._eps)
        return float(-np.mean(y * np.log(yh) + (1.0 - y) * np.log(1.0 - yh)))

    def grad(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        yh = np.clip(y_hat, self._eps, 1.0 - self._eps)
        return (-(y / yh) + (1.0 - y) / (1.0 - yh)) / y_hat.size


class VAELoss:
    """
    ELBO para VAE:  L = BCE(x̂, x)  +  β · KL(q(z|x) || N(0,I))

    El gradiente del término KL lo maneja VAE.backward() internamente.
    Esta clase solo expone el valor escalar total y el gradiente de reconstrucción.

    KL en forma cerrada:  KL = -½ · mean(1 + logvar - μ² - exp(logvar))
    """

    _bce = BCE()

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta
        self.last_recon: float = 0.0
        self.last_kl:    float = 0.0

    def __call__(self, y_hat: np.ndarray, y: np.ndarray,
                 mu: np.ndarray, logvar: np.ndarray) -> float:
        self.last_recon = self._bce(y_hat, y)
        self.last_kl    = float(-0.5 * np.mean(1.0 + logvar - mu ** 2 - np.exp(logvar)))
        return self.last_recon + self.beta * self.last_kl

    def grad(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solo el gradiente de reconstrucción — el KL se añade en VAE.backward()."""
        return self._bce.grad(y_hat, y)
