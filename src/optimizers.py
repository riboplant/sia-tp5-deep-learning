import numpy as np


class SGD:
    """SGD con momentum: v = momentum*v - lr*grad  ;  p += v"""

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self._v: dict[int, np.ndarray] = {}

    def step(self, params: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for i, (p, g) in enumerate(params):
            if i not in self._v:
                self._v[i] = np.zeros_like(p)
            self._v[i] = self.momentum * self._v[i] - self.lr * g
            p += self._v[i]


class Adam:
    """Adam: momentos adaptativos de primer y segundo orden."""

    def __init__(self, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, params: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(params):
            if i not in self._m:
                self._m[i] = np.zeros_like(p)
                self._v[i] = np.zeros_like(p)
            self._m[i] = self.b1 * self._m[i] + (1.0 - self.b1) * g
            self._v[i] = self.b2 * self._v[i] + (1.0 - self.b2) * g ** 2
            m_hat = self._m[i] / (1.0 - self.b1 ** self.t)
            v_hat = self._v[i] / (1.0 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
