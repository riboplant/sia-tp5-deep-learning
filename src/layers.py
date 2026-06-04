import numpy as np


class Dense:
    """
    Capa fully-connected: out = x @ W.T + b

    Forward : (B, in_size)  -> (B, out_size)
    Backward: (B, out_size) -> (B, in_size)  y acumula dW, db
    """

    def __init__(self, in_size: int, out_size: int) -> None:
        limit = np.sqrt(6.0 / (in_size + out_size))
        self.W = np.random.uniform(-limit, limit, (out_size, in_size))
        self.b = np.zeros(out_size)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W.T + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.dW = grad.T @ self._x   # (out, in) — suma sobre el batch
        self.db = grad.sum(axis=0)   # (out,)
        return grad @ self.W         # (B, in) — propaga hacia la capa anterior

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.W, self.dW), (self.b, self.db)]


class Tanh:
    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._out ** 2)

    def params(self) -> list:
        return []


class Sigmoid:
    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Versión numéricamente estable
        self._out = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._out * (1.0 - self._out)

    def params(self) -> list:
        return []
