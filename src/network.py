import numpy as np


class Network:
    """Cadena secuencial de capas con forward y backward."""

    def __init__(self, layers: list) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        p = []
        for layer in self.layers:
            p.extend(layer.params())
        return p
