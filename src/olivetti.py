import numpy as np
from sklearn.datasets import fetch_olivetti_faces


def load_olivetti(size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga Olivetti Faces (400 caras, 40 personas × 10 fotos).
    Redimensiona a size×size por block-averaging y aplana a (400, size²).
    Retorna (data, labels) con data en [0, 1].
    """
    dataset = fetch_olivetti_faces(download_if_missing=True, shuffle=False)
    images = dataset.images  # (400, 64, 64) float64 en [0, 1]
    labels = dataset.target  # (400,) int en [0, 39]
    if size != 64:
        images = _resize_batch(images, size)
    return images.reshape(len(images), -1).astype(np.float64), labels


def _resize_batch(images: np.ndarray, size: int) -> np.ndarray:
    """Downscale (N, H, H) → (N, size, size) por block-averaging."""
    n, h, _ = images.shape
    f = h // size
    return images.reshape(n, size, f, size, f).mean(axis=(2, 4))
