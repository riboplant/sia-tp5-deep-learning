import numpy as np


# ---------------------------------------------------------------------------
# Helpers im2col / col2im para Conv2D
# ---------------------------------------------------------------------------

def _im2col(x: np.ndarray, kH: int, kW: int, padding: int) -> np.ndarray:
    """
    (N,C,H,W) → (N*H_out*W_out, C*kH*kW)

    Usa as_strided para crear una vista zero-copy de los patches — sin loops
    de Python sobre las posiciones del kernel.
    """
    N, C, H, W = x.shape
    H_out = H + 2 * padding - kH + 1
    W_out = W + 2 * padding - kW + 1
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    # Strides del array paddeado: (N, C, H_pad, W_pad)
    sN, sC, sH, sW = x.strides
    # Vista de shape (N, C, H_out, W_out, kH, kW) sin copiar datos
    patches = np.lib.stride_tricks.as_strided(
        x,
        shape=(N, C, H_out, W_out, kH, kW),
        strides=(sN, sC, sH, sW, sH, sW),
    )
    # (N, C, H_out, W_out, kH, kW) → (N*H_out*W_out, C*kH*kW)  — .copy() materializa la vista
    return patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * H_out * W_out, C * kH * kW).copy()


def _col2im(col: np.ndarray, x_shape: tuple, kH: int, kW: int, padding: int) -> np.ndarray:
    """
    (N*H_out*W_out, C*kH*kW) → (N,C,H,W)

    Los loops son sobre kH×kW posiciones (9 para kernel 3×3) — aceptable.
    """
    N, C, H, W = x_shape
    H_out = H + 2 * padding - kH + 1
    W_out = W + 2 * padding - kW + 1
    col_r = col.reshape(N, H_out, W_out, C, kH, kW).transpose(0, 3, 4, 5, 1, 2)
    x_pad = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=col.dtype)
    for i in range(kH):
        for j in range(kW):
            x_pad[:, :, i:i + H_out, j:j + W_out] += col_r[:, :, i, j, :, :]
    if padding == 0:
        return x_pad
    return x_pad[:, :, padding:-padding, padding:-padding]


# ---------------------------------------------------------------------------
# Capas CNN
# ---------------------------------------------------------------------------

class Conv2D:
    """
    Convolución 2D: (N,C_in,H,W) → (N,C_out,H_out,W_out)

    kernel_size×kernel_size, padding same por defecto (pad = kernel_size//2).
    Inicialización Xavier uniforme.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int | None = None) -> None:
        self.C_in  = in_channels
        self.C_out = out_channels
        self.kH = self.kW = kernel_size
        self.padding = kernel_size // 2 if padding is None else padding

        fan_in  = in_channels  * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self.W  = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size))
        self.b  = np.zeros(out_channels)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._col: np.ndarray | None = None
        self._x_shape: tuple | None  = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        H_out = H + 2 * self.padding - self.kH + 1
        W_out = W + 2 * self.padding - self.kW + 1
        self._x_shape = x.shape
        self._col = _im2col(x, self.kH, self.kW, self.padding)      # (N*H_out*W_out, C*kH*kW)
        W_mat = self.W.reshape(self.C_out, -1)                        # (C_out, C*kH*kW)
        out = (self._col @ W_mat.T + self.b)                          # (N*H_out*W_out, C_out)
        return out.reshape(N, H_out, W_out, self.C_out).transpose(0, 3, 1, 2)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        N, C_out, H_out, W_out = grad.shape
        grad_mat = grad.transpose(0, 2, 3, 1).reshape(-1, C_out)      # (N*H_out*W_out, C_out)
        self.dW = (grad_mat.T @ self._col).reshape(self.W.shape)
        self.db = grad_mat.sum(axis=0)
        grad_col = grad_mat @ self.W.reshape(self.C_out, -1)           # (N*H_out*W_out, C*kH*kW)
        return _col2im(grad_col, self._x_shape, self.kH, self.kW, self.padding)

    def params(self) -> list:
        return [(self.W, self.dW), (self.b, self.db)]


class MaxPool2D:
    """
    Max pooling 2D (non-overlapping, stride=kernel_size).
    Requiere H y W divisibles por kernel_size.
    """

    def __init__(self, kernel_size: int = 2) -> None:
        self.k = kernel_size
        self._mask:    np.ndarray | None = None
        self._x_shape: tuple | None      = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        k = self.k
        H_out, W_out = H // k, W // k
        self._x_shape = x.shape
        x_r = x.reshape(N, C, H_out, k, W_out, k)
        out = x_r.max(axis=(3, 5))
        # Máscara: 1 en la posición máxima, distribuida en caso de empate
        mask = (x_r == out[:, :, :, np.newaxis, :, np.newaxis]).astype(float)
        counts = mask.sum(axis=(3, 5), keepdims=True)
        self._mask = mask / counts
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad_exp = grad[:, :, :, np.newaxis, :, np.newaxis]
        return (self._mask * grad_exp).reshape(self._x_shape)

    def params(self) -> list:
        return []


class Upsample2D:
    """
    Upsampling nearest-neighbor: (N,C,H,W) → (N,C,H*scale,W*scale).
    Backward: suma los gradientes de cada bloque scale×scale.
    """

    def __init__(self, scale: int = 2) -> None:
        self.scale = scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(x, self.scale, axis=2), self.scale, axis=3)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        N, C, H, W = grad.shape
        s = self.scale
        return grad.reshape(N, C, H // s, s, W // s, s).sum(axis=(3, 5))

    def params(self) -> list:
        return []


class Flatten:
    """(N, *) → (N, prod(*))  con backward que restaura la forma original."""

    def __init__(self) -> None:
        self._shape: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self._shape)

    def params(self) -> list:
        return []


class Reshape:
    """(N, -1) → (N, *target_shape)  — útil para pasar de Dense a capas CNN."""

    def __init__(self, target_shape: tuple) -> None:
        self.target_shape = target_shape
        self._n: int | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._n = x.shape[0]
        return x.reshape(self._n, *self.target_shape)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self._n, -1)

    def params(self) -> list:
        return []


# ---------------------------------------------------------------------------
# Capas Dense y activaciones
# ---------------------------------------------------------------------------

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
