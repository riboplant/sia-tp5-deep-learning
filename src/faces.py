from itertools import product

import numpy as np

H, W = 9, 9  # grid de cada cara: 81 píxeles

# ---------------------------------------------------------------------------
# Componentes de la cara (listas de (fila, col) a poner en 1)
# ---------------------------------------------------------------------------

_BASE = [
    # contorno oval
    (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 6),
    (2, 2), (2, 6),
    (3, 2), (3, 6),
    (4, 2), (4, 6),
    (5, 2), (5, 6),
    (6, 2), (6, 6),
    (7, 3), (7, 4), (7, 5),
    # nariz
    (4, 4),
]

_EYEBROWS = {
    "neutral": [(1, 3), (1, 5)],              # puntos separados sobre cada ojo
    "raised":  [(1, 3), (1, 4), (1, 5)],      # barra continua — cejas levantadas
    "angry":   [(2, 3), (2, 5)],              # un nivel más abajo, hacia la nariz — fruncidas
}

_EYES = {
    "open":   [(2, 3), (2, 5), (3, 3), (3, 5)],   # 2×1 por ojo
    "half":   [(3, 3), (3, 5)],                    # solo fila inferior — semicerrados
    "closed": [(3, 3), (3, 4), (3, 5)],            # barra horizontal — cerrados
}

_MOUTHS = {
    "smile":   [(5, 3), (5, 5), (6, 4)],     # comisuras arriba, centro abajo — sonrisa
    "neutral": [(5, 3), (5, 4), (5, 5)],      # línea plana
    "frown":   [(5, 4), (6, 3), (6, 5)],      # centro arriba, comisuras abajo — tristeza
}

_ACCESSORIES = {
    "none":    [],
    "glasses": [(1, 3), (1, 4), (1, 5), (2, 4)],  # barra superior + puente central
}

MOUTH_VALUES     = ["smile", "neutral", "frown"]
EYE_VALUES       = ["open", "half", "closed"]
EYEBROW_VALUES   = ["neutral", "raised", "angry"]
ACCESSORY_VALUES = ["none", "glasses"]


def make_face(mouth: str, eyes: str, eyebrows: str, accessory: str) -> np.ndarray:
    face = np.zeros((H, W), dtype=float)
    for pixels in (_BASE, _EYEBROWS[eyebrows], _EYES[eyes],
                   _MOUTHS[mouth], _ACCESSORIES[accessory]):
        for r, c in pixels:
            face[r, c] = 1.0
    return face.flatten()


def load_faces() -> tuple[np.ndarray, list[str], list[dict]]:
    """
    Genera las 54 caras (3 bocas × 3 ojos × 3 cejas × 2 accesorios).

    Returns
    -------
    data   : (54, 81) array binario
    labels : lista de strings "boca_ojos_cejas_accesorio"
    attrs  : lista de dicts con los valores de cada atributo por cara
    """
    data, labels, attrs = [], [], []
    combos = list(product(MOUTH_VALUES, EYE_VALUES, EYEBROW_VALUES, ACCESSORY_VALUES))
    for mouth, eyes, eyebrows, accessory in combos:
        data.append(make_face(mouth, eyes, eyebrows, accessory))
        labels.append(f"{mouth}_{eyes}_{eyebrows}_{accessory}")
        attrs.append({"mouth": mouth, "eyes": eyes,
                      "eyebrows": eyebrows, "accessory": accessory})
    return np.array(data), labels, attrs
