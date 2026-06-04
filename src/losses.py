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
