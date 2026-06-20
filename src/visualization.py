import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_RESULTS_DIR: str = "results"


def set_results_dir(path: str) -> None:
    global _RESULTS_DIR
    _RESULTS_DIR = path


def _save(filename: str) -> None:
    import os
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    plt.savefig(f"{_RESULTS_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [guardado] {_RESULTS_DIR}/{filename}")


def plot_chars(
    chars: list[np.ndarray],
    labels: list[str] | None = None,
    cols: int = 8,
    title: str | None = None,
    threshold: float | None = None,
    filename: str | None = None,
    shape: tuple[int, int] = (7, 5),
) -> None:
    imgs = [(c > threshold).astype(float) if threshold is not None else c for c in chars]
    n = len(imgs)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.6))
    axes = np.array(axes).flatten()
    for i, img in enumerate(imgs):
        axes[i].imshow(img.reshape(shape), cmap="binary", vmin=0, vmax=1)
        if labels:
            axes[i].set_title(labels[i], fontsize=9)
        axes[i].axis("off")
    for j in range(len(imgs), len(axes)):
        axes[j].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_training(results: dict, title: str, filename: str | None = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    colors = plt.cm.tab10.colors

    for i, (name, r) in enumerate(results.items()):
        h = r["history"]
        c = colors[i % len(colors)]
        ax1.semilogy(h["loss"], label=name, color=c, alpha=0.85)
        ax2.plot(h["max_px_err"], label=name, color=c, alpha=0.85)
        if h["converged_at"]:
            ax1.axvline(h["converged_at"], color=c, linestyle=":", alpha=0.6)
            ax2.axvline(h["converged_at"], color=c, linestyle=":", alpha=0.6)

    ax1.set(title="Pérdida (log)", xlabel="Época")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axhline(1, color="red", linestyle="--", linewidth=1.5, label="Objetivo (≤1 px)")
    ax2.set(title="Error máximo en píxeles", xlabel="Época")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if filename:
        _save(filename)

    print(f"\n{'Configuración':<30} {'Convergencia':<15} {'Loss final':<15} {'Max err final'}")
    print("-" * 75)
    for name, r in results.items():
        h = r["history"]
        conv = str(h["converged_at"]) if h["converged_at"] else "No convergió"
        print(f"{name:<30} {conv:<15} {h['loss'][-1]:<15.5f} {h['max_px_err'][-1]:.0f}")


def plot_latent_space(latent: np.ndarray, labels: list[str], filename: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(latent[:, 0], latent[:, 1], c=range(len(labels)), cmap="tab20", s=80, zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, (latent[i, 0], latent[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=10)
    ax.set_title("Espacio latente 2D", fontsize=13)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        _save(filename)


# ---------------------------------------------------------------------------
# Visualizaciones adicionales del mejor modelo
# ---------------------------------------------------------------------------

def plot_latent_thumbnails(
    latent: np.ndarray,
    data: np.ndarray,
    labels: list[str],
    filename: str | None = None,
) -> None:
    """Espacio latente con la imagen del carácter en lugar de un punto."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    margin = 1.0
    fig, ax = plt.subplots(figsize=(14, 11))

    for i, (z, char) in enumerate(zip(latent, data)):
        img = char.reshape(7, 5)
        imgbox = OffsetImage(img, zoom=2.8, cmap="binary")
        imgbox.image.axes = ax
        ab = AnnotationBbox(
            imgbox, (z[0], z[1]),
            frameon=True,
            bboxprops=dict(edgecolor="steelblue", linewidth=0.8, boxstyle="round,pad=0.1"),
            pad=0.1,
        )
        ax.add_artist(ab)
        ax.text(z[0], z[1] - 0.55, labels[i], ha="center", va="top", fontsize=7, color="dimgray")

    ax.set_xlim(latent[:, 0].min() - margin, latent[:, 0].max() + margin)
    ax.set_ylim(latent[:, 1].min() - margin, latent[:, 1].max() + margin)
    ax.set_title("Espacio latente 2D — miniaturas de caracteres", fontsize=13)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.grid(True, alpha=0.25, zorder=0)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_continuous_reconstruction(
    data: np.ndarray,
    reconstructed: np.ndarray,
    labels: list[str],
    filename: str | None = None,
) -> None:
    """
    Para cada carácter muestra 3 paneles en una fila:
      original binario | salida continua (escala de grises) | mapa de confianza
    La confianza es |output - 0.5| * 2: blanco = seguro, negro = dudoso.
    """
    n = len(data)
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows * 3, cols, figsize=(cols * 1.4, rows * 4.5))
    axes = np.array(axes).reshape(rows * 3, cols)

    for i in range(n):
        row_block = (i // cols) * 3
        col = i % cols

        orig = data[i].reshape(7, 5)
        cont = reconstructed[i].reshape(7, 5)
        conf = np.abs(cont - 0.5) * 2  # 1 = seguro, 0 = dudoso

        axes[row_block,     col].imshow(orig, cmap="binary", vmin=0, vmax=1)
        axes[row_block + 1, col].imshow(cont, cmap="binary", vmin=0, vmax=1)
        axes[row_block + 2, col].imshow(conf, cmap="RdYlGn", vmin=0, vmax=1)

        axes[row_block, col].set_title(labels[i], fontsize=8)
        for r in range(3):
            axes[row_block + r, col].axis("off")

    # Ocultar celdas vacías
    for i in range(n, rows * cols):
        row_block = (i // cols) * 3
        col = i % cols
        for r in range(3):
            axes[row_block + r, col].axis("off")

    # Leyenda de filas
    for r, label in enumerate(["Original", "Continuo", "Confianza"]):
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=40, va="center")

    fig.suptitle(
        "Reconstrucción continua  |  Verde = seguro (≥0.5 de margen)  |  Rojo = dudoso (cerca de 0.5)",
        fontsize=11,
    )
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_pixel_error_detail(
    data: np.ndarray,
    reconstructed: np.ndarray,
    labels: list[str],
    filename: str | None = None,
) -> None:
    """
    Para cada carácter con al menos 1 pixel incorrecto muestra:
      original | reconstruido binarizado | mapa de diferencia (píxel erróneo en rojo)
    """
    errors_per_char = np.abs(
        (reconstructed > 0.5).astype(float) - data
    ).sum(axis=1)
    bad = [(i, labels[i]) for i in range(len(data)) if errors_per_char[i] > 0]

    if not bad:
        print("  Todos los caracteres se reconstruyeron perfectamente (0 px de error).")
        return

    n = len(bad)
    fig, axes = plt.subplots(n, 3, figsize=(5, n * 2.0))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (i, label) in enumerate(bad):
        orig = data[i].reshape(7, 5)
        recon = (reconstructed[i].reshape(7, 5) > 0.5).astype(float)
        diff = np.abs(orig - recon)

        # Imagen RGB para el mapa de error: fondo blanco, pixeles correctos en gris claro,
        # pixeles erróneos en rojo intenso
        diff_rgb = np.ones((7, 5, 3))
        diff_rgb[:, :, 0] = 1.0
        diff_rgb[:, :, 1] = 1.0 - diff          # quita verde en error
        diff_rgb[:, :, 2] = 1.0 - diff          # quita azul en error

        axes[row, 0].imshow(orig, cmap="binary", vmin=0, vmax=1)
        axes[row, 0].set_title(f"'{label}' original", fontsize=9)
        axes[row, 1].imshow(recon, cmap="binary", vmin=0, vmax=1)
        axes[row, 1].set_title(f"'{label}' reconstruido", fontsize=9)
        axes[row, 2].imshow(diff_rgb)
        axes[row, 2].set_title(f"Error: {int(errors_per_char[i])} px (rojo)", fontsize=9)

        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle("Detalle de error por píxel", fontsize=12)
    plt.tight_layout()
    if filename:
        _save(filename)


# ---------------------------------------------------------------------------
# Visualizaciones para Denoising Autoencoder
# ---------------------------------------------------------------------------

def plot_noise_levels(
    data: np.ndarray,
    noise_fn,
    levels: list[float],
    labels: list[str],
    noise_name: str,
    filename: str | None = None,
) -> None:
    """Muestra los primeros 8 caracteres corrompidos a cada nivel de ruido."""
    rng = np.random.default_rng(0)
    n_chars = min(8, len(data))
    n_levels = len(levels)

    fig, axes = plt.subplots(n_levels + 1, n_chars, figsize=(n_chars * 1.2, (n_levels + 1) * 1.6))

    for col in range(n_chars):
        axes[0, col].imshow(data[col].reshape(7, 5), cmap="binary", vmin=0, vmax=1)
        axes[0, col].set_title(labels[col], fontsize=8)
        axes[0, col].axis("off")

    for row, p in enumerate(levels, start=1):
        noisy = noise_fn(data[:n_chars], p, rng)
        for col in range(n_chars):
            axes[row, col].imshow(noisy[col].reshape(7, 5), cmap="binary", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"p={p}", fontsize=8, rotation=0, labelpad=30, va="center")

    axes[0, 0].set_ylabel("Original", fontsize=8, rotation=0, labelpad=30, va="center")
    fig.suptitle(f"Niveles de ruido — {noise_name}", fontsize=12)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_denoising_summary(
    results_by_level: dict[float, dict],
    noise_name: str,
    arch_label: str,
    filename: str | None = None,
) -> None:
    """Curvas de loss y max px error para cada nivel de ruido, en un mismo plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(results_by_level)))

    for color, (p, r) in zip(colors, results_by_level.items()):
        h = r["history"]
        label = f"p={p}"
        ax1.semilogy(h["loss"], label=label, color=color, alpha=0.85)
        ax2.plot(h["max_px_err"], label=label, color=color, alpha=0.85)
        if h["converged_at"]:
            ax1.axvline(h["converged_at"], color=color, linestyle=":", alpha=0.5)
            ax2.axvline(h["converged_at"], color=color, linestyle=":", alpha=0.5)

    ax1.set(title="Pérdida (log)", xlabel="Época")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(1, color="red", linestyle="--", linewidth=1.5, label="Objetivo (≤1 px)")
    ax2.set(title="Error máximo en píxeles", xlabel="Época")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Denoising AE — {noise_name} | {arch_label}", fontsize=13)
    plt.tight_layout()
    if filename:
        _save(filename)

    print(f"\n{'Nivel':<10} {'Convergencia':<15} {'Loss final':<15} {'Max err final'}")
    print("-" * 55)
    for p, r in results_by_level.items():
        h = r["history"]
        conv = str(h["converged_at"]) if h["converged_at"] else "No convergió"
        print(f"p={p:<8} {conv:<15} {h['loss'][-1]:<15.5f} {h['max_px_err'][-1]:.0f}")


def plot_denoising_comparison(
    sp_results: dict[float, dict],
    mask_results: dict[float, dict],
    filename: str | None = None,
) -> None:
    """Compara convergencia entre Salt & Pepper y Masking para cada nivel de ruido."""
    levels = sorted(sp_results.keys())

    sp_conv  = [sp_results[p]["history"]["converged_at"] or float("nan") for p in levels]
    mask_conv = [mask_results[p]["history"]["converged_at"] or float("nan") for p in levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(levels))
    w = 0.35

    bars1 = ax1.bar(x - w / 2, sp_conv,   w, label="Salt & Pepper", color="steelblue",  alpha=0.85)
    bars2 = ax1.bar(x + w / 2, mask_conv, w, label="Masking",        color="darkorange", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"p={p}" for p in levels])
    ax1.set_ylabel("Época de convergencia")
    ax1.set_title("Velocidad de convergencia por nivel de ruido")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # Marcar barras de no-convergencia
    for bars, results in [(bars1, sp_results), (bars2, mask_results)]:
        for bar, p in zip(bars, levels):
            if results[p]["history"]["converged_at"] is None:
                ax1.text(bar.get_x() + bar.get_width() / 2, 2000,
                         "✗", ha="center", va="bottom", fontsize=12, color="red")

    sp_err   = [sp_results[p]["history"]["max_px_err"][-1]  for p in levels]
    mask_err = [mask_results[p]["history"]["max_px_err"][-1] for p in levels]

    ax2.plot(levels, sp_err,   "o-", label="Salt & Pepper", color="steelblue",  linewidth=2)
    ax2.plot(levels, mask_err, "s-", label="Masking",        color="darkorange", linewidth=2)
    ax2.axhline(1, color="red", linestyle="--", linewidth=1.5, label="Objetivo (≤1 px)")
    ax2.set_xlabel("Nivel de ruido p")
    ax2.set_ylabel("Max px error final")
    ax2.set_title("Error final vs nivel de ruido")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Salt & Pepper vs Masking — comparación directa", fontsize=13)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_denoising_reconstructions(
    model,
    data: np.ndarray,
    noise_fn,
    levels: list[float],
    labels: list[str],
    noise_name: str,
    n_chars: int = 8,
    filename: str | None = None,
) -> None:
    """Para cada nivel de ruido muestra: original | ruidoso | reconstruido."""
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(len(levels) * 3, n_chars, figsize=(n_chars * 1.2, len(levels) * 4.8))

    for block, p in enumerate(levels):
        noisy = noise_fn(data[:n_chars], p, rng)
        recon = model.forward(noisy)

        for col in range(n_chars):
            r0, r1, r2 = block * 3, block * 3 + 1, block * 3 + 2
            axes[r0, col].imshow(data[col].reshape(7, 5), cmap="binary", vmin=0, vmax=1)
            axes[r1, col].imshow(noisy[col].reshape(7, 5), cmap="binary", vmin=0, vmax=1)
            axes[r2, col].imshow((recon[col] > 0.5).reshape(7, 5), cmap="binary", vmin=0, vmax=1)
            if block == 0:
                axes[r0, col].set_title(labels[col], fontsize=8)
            for r in (r0, r1, r2):
                axes[r, col].axis("off")

        axes[block * 3,     0].set_ylabel("Original",     fontsize=7, rotation=0, labelpad=35, va="center")
        axes[block * 3 + 1, 0].set_ylabel(f"Ruido p={p}", fontsize=7, rotation=0, labelpad=35, va="center")
        axes[block * 3 + 2, 0].set_ylabel("Reconstruido", fontsize=7, rotation=0, labelpad=35, va="center")

    fig.suptitle(f"Denoising — {noise_name}: original / ruidoso / reconstruido", fontsize=11)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_generative_grid_nn(
    model,
    latent: np.ndarray,
    labels: list[str],
    n: int = 12,
    filename: str | None = None,
) -> None:
    """
    Grilla n×n en el espacio latente. Cada celda muestra el carácter generado
    y el label del vecino más cercano del dataset de entrenamiento.
    """
    z1_range = np.linspace(latent[:, 0].min() - 0.5, latent[:, 0].max() + 0.5, n)
    z2_range = np.linspace(latent[:, 1].max() + 0.5, latent[:, 1].min() - 0.5, n)

    fig, axes = plt.subplots(n, n, figsize=(n * 1.1, n * 1.3))

    for row, z2 in enumerate(z2_range):
        for col, z1 in enumerate(z1_range):
            z = np.array([[z1, z2]])
            gen = (model.decode(z).flatten() > 0.5).astype(float)

            dists = np.linalg.norm(latent - np.array([z1, z2]), axis=1)
            nn_label = labels[int(np.argmin(dists))]

            axes[row, col].imshow(gen.reshape(7, 5), cmap="binary", vmin=0, vmax=1)
            axes[row, col].set_title(nn_label, fontsize=6, pad=1)
            axes[row, col].axis("off")

    fig.suptitle(
        f"Grilla generativa {n}×{n} — etiqueta = vecino más cercano en espacio latente",
        fontsize=11,
    )
    plt.tight_layout()
    if filename:
        _save(filename)


# ---------------------------------------------------------------------------
# Visualizaciones para VAE
# ---------------------------------------------------------------------------

_ATTR_COLORS = {
    "mouth":     {"smile": "tab:green",  "neutral": "tab:blue",   "frown": "tab:red"},
    "eyes":      {"open":  "tab:blue",   "half":    "tab:orange",  "closed": "tab:red"},
    "eyebrows":  {"neutral": "tab:blue", "raised":  "tab:green",  "angry": "tab:red"},
    "accessory": {"none":  "tab:blue",   "glasses": "tab:orange"},
}


def plot_vae_training(history: dict, filename: str | None = None) -> None:
    """Curvas de loss total, reconstrucción y KL durante el entrenamiento del VAE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.semilogy(history["loss"],  label="Total",        color="black",      alpha=0.9)
    ax1.semilogy(history["recon"], label="Reconstrucción", color="steelblue", alpha=0.85)
    ax1.semilogy(np.abs(history["kl"]), label="|KL|",    color="darkorange", alpha=0.85)
    if history["converged_at"]:
        ax1.axvline(history["converged_at"], color="gray", linestyle=":", alpha=0.7)
    ax1.set(title="Pérdida (log)", xlabel="Época")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["max_px_err"], color="steelblue", alpha=0.85)
    ax2.axhline(1, color="red", linestyle="--", linewidth=1.5, label="Objetivo (≤1 px)")
    if history["converged_at"]:
        ax2.axvline(history["converged_at"], color="gray", linestyle=":", alpha=0.7)
    ax2.set(title="Error máximo en píxeles", xlabel="Época")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Entrenamiento VAE", fontsize=13)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_latent_by_attribute(
    latent: np.ndarray,
    attrs: list[dict],
    filename: str | None = None,
) -> None:
    """4 scatter plots del espacio latente, uno por atributo, con colores por valor."""
    attr_names = ["mouth", "eyes", "eyebrows", "accessory"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for ax, attr in zip(axes, attr_names):
        color_map = _ATTR_COLORS[attr]
        for val, color in color_map.items():
            mask = [i for i, a in enumerate(attrs) if a[attr] == val]
            ax.scatter(latent[mask, 0], latent[mask, 1],
                       c=color, label=val, s=60, alpha=0.85, zorder=3)
        ax.set_title(f"Atributo: {attr}", fontsize=11)
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Espacio latente VAE coloreado por atributo", fontsize=13)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_vae_samples(
    model,
    n: int = 5,
    rows: int | None = None,
    cols: int | None = None,
    shape: tuple[int, int] = (9, 9),
    threshold: float | None = 0.5,
    filename: str | None = None,
) -> None:
    """
    Samplea z ~ N(0,I) y decodifica.

    Si se pasan rows/cols, genera exactamente rows×cols muestras en esa grilla.
    Si solo se pasa n, genera una grilla n×n (comportamiento original).
    threshold=0.5 binariza la imagen; threshold=None muestra valores continuos en escala de grises.
    """
    r = rows if rows is not None else n
    c = cols if cols is not None else n
    rng = np.random.default_rng(0)
    z_samples = rng.standard_normal((r * c, model.latent_dim))
    generated = model.decode(z_samples)

    fig, axes = plt.subplots(r, c, figsize=(c * 1.4, r * 1.6))
    for i, ax in enumerate(np.array(axes).flatten()):
        img = (generated[i] > threshold).reshape(shape) if threshold is not None \
              else generated[i].reshape(shape)
        cmap = "binary" if threshold is not None else "gray"
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle(f"Muestras del prior N(0,I) — {r}×{c} caras generadas", fontsize=12)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_face_interpolation(
    model,
    latent: np.ndarray,
    idx_a: int,
    idx_b: int,
    label_a: str,
    label_b: str,
    n_steps: int = 9,
    shape: tuple[int, int] = (9, 9),
    threshold: float | None = 0.5,
    filename: str | None = None,
) -> None:
    """Interpolación lineal entre dos caras en el espacio latente.
    threshold=None muestra valores continuos en escala de grises."""
    alphas = np.linspace(0, 1, n_steps)
    interp = [(1 - a) * latent[idx_a] + a * latent[idx_b] for a in alphas]
    generated = [model.decode(z.reshape(1, -1)).flatten() for z in interp]

    cmap = "binary" if threshold is not None else "gray"
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.4, 2.0))
    for i, (ax, img) in enumerate(zip(axes, generated)):
        display = (img > threshold).reshape(shape) if threshold is not None else img.reshape(shape)
        ax.imshow(display, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{alphas[i]:.2f}", fontsize=7)
        ax.axis("off")

    fig.suptitle(f"Interpolación: '{label_a}' → '{label_b}'", fontsize=11)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_ae_vs_vae_latent(
    latent_ae: np.ndarray,
    latent_vae: np.ndarray,
    logvar_vae: np.ndarray,
    attrs: list[dict],
    filename: str | None = None,
) -> None:
    """
    Compara la estructura del espacio latente de AE y VAE.
    AE: puntos fijos aislados + bounding box de muestreo.
    VAE: puntos μ con círculos de radio σ + círculos del prior N(0,1).
    Colorea por boca (el atributo más separable).
    """
    mouth_colors = {"smile": "tab:green", "neutral": "tab:blue", "frown": "tab:red"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # --- AE ---
    for val, color in mouth_colors.items():
        mask = [i for i, a in enumerate(attrs) if a["mouth"] == val]
        ax1.scatter(latent_ae[mask, 0], latent_ae[mask, 1],
                    c=color, label=val, s=70, zorder=3)
    # Bounding box = región "muestreable" para el AE
    from matplotlib.patches import Rectangle
    z1lo, z1hi = latent_ae[:, 0].min(), latent_ae[:, 0].max()
    z2lo, z2hi = latent_ae[:, 1].min(), latent_ae[:, 1].max()
    ax1.add_patch(Rectangle((z1lo, z2lo), z1hi - z1lo, z2hi - z2lo,
                             fill=False, edgecolor="darkorange", linewidth=2,
                             linestyle="--", label="Región muestreable (bounding box)"))
    ax1.set_title("AE — puntos fijos, sin regularización", fontsize=11)
    ax1.set_xlabel("z₁"); ax1.set_ylabel("z₂")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # --- VAE ---
    sigma = np.exp(0.5 * logvar_vae).mean(axis=1)  # σ escalar por muestra
    for val, color in mouth_colors.items():
        mask = [i for i, a in enumerate(attrs) if a["mouth"] == val]
        ax2.scatter(latent_vae[mask, 0], latent_vae[mask, 1],
                    c=color, label=val, s=70, zorder=4)
        for i in mask:
            ax2.add_patch(plt.Circle((latent_vae[i, 0], latent_vae[i, 1]),
                                      sigma[i], color=color, alpha=0.12, zorder=2))
    # Prior N(0,1)
    for r, ls, lbl in [(1, "--", "Prior 1σ"), (2, ":", "Prior 2σ")]:
        ax2.add_patch(plt.Circle((0, 0), r, fill=False, edgecolor="gray",
                                  linewidth=1.5, linestyle=ls, label=lbl, zorder=1))
    ax2.set_xlim(-3.2, 3.2); ax2.set_ylim(-3.2, 3.2)
    ax2.set_aspect("equal")
    ax2.set_title("VAE — distribuciones μ±σ solapadas, centradas en prior", fontsize=11)
    ax2.set_xlabel("z₁"); ax2.set_ylabel("z₂")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.suptitle("Estructura del espacio latente: AE vs VAE", fontsize=13)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_ae_vs_vae_grid(
    ae_model,
    vae_model,
    latent_ae: np.ndarray,
    latent_vae: np.ndarray,
    shape: tuple[int, int],
    n: int = 10,
    filename: str | None = None,
) -> None:
    """
    Decodifica una grilla n×n para AE y VAE lado a lado.
    El AE usa el bounding box de sus puntos de entrenamiento.
    El VAE usa [-2.5, 2.5]² (rango natural del prior N(0,1)).
    Los artefactos del AE en zonas sin datos son evidentes.
    """
    margin = 0.3
    z1_ae = np.linspace(latent_ae[:, 0].min() - margin, latent_ae[:, 0].max() + margin, n)
    z2_ae = np.linspace(latent_ae[:, 1].max() + margin, latent_ae[:, 1].min() - margin, n)
    z1_vae = np.linspace(-2.5, 2.5, n)
    z2_vae = np.linspace(2.5, -2.5, n)

    fig, axes = plt.subplots(n, n * 2 + 1, figsize=(n * 2.6, n * 1.4))

    for row in range(n):
        for col in range(n):
            # AE
            z = np.array([[z1_ae[col], z2_ae[row]]])
            gen_ae = (ae_model.decode(z).flatten() > 0.5).astype(float)
            axes[row, col].imshow(gen_ae.reshape(shape), cmap="binary", vmin=0, vmax=1)
            axes[row, col].axis("off")
            # Marcar si hay un punto de entrenamiento cerca
            dists = np.linalg.norm(latent_ae - np.array([z1_ae[col], z2_ae[row]]), axis=1)
            if dists.min() < (z1_ae[1] - z1_ae[0]) * 1.5:
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor("lime"); spine.set_linewidth(2)
                axes[row, col].set_visible(True)

            # Separador visual (columna central vacía)
            axes[row, n].axis("off")

            # VAE
            z = np.array([[z1_vae[col], z2_vae[row]]])
            gen_vae = (vae_model.decode(z).flatten() > 0.5).astype(float)
            axes[row, col + n + 1].imshow(gen_vae.reshape(shape), cmap="binary", vmin=0, vmax=1)
            axes[row, col + n + 1].axis("off")

    fig.text(0.25, 1.01, f"AE — bounding box (borde verde = cerca de dato de entrenamiento)",
             ha="center", fontsize=10)
    fig.text(0.75, 1.01, f"VAE — prior N(0,1)  [-2.5, 2.5]²",
             ha="center", fontsize=10)
    fig.suptitle(f"Grid decode {n}×{n}: AE vs VAE", fontsize=13, y=1.04)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_ae_vs_vae_interpolation(
    ae_model,
    vae_model,
    latent_ae: np.ndarray,
    latent_vae: np.ndarray,
    attrs: list[dict],
    shape: tuple[int, int],
    n_steps: int = 9,
    filename: str | None = None,
) -> None:
    """
    Interpola entre una cara feliz y una triste para AE y VAE.
    Muestra si el AE cruza una zona muerta (artefactos) y el VAE no.
    Incluye debajo un mini-mapa del espacio latente con la trayectoria.
    """
    def find(mouth, eyes, eyebrows):
        for i, a in enumerate(attrs):
            if a["mouth"] == mouth and a["eyes"] == eyes and a["eyebrows"] == eyebrows:
                return i
        return 0

    idx_a = find("smile", "open", "raised")
    idx_b = find("frown", "closed", "angry")
    alphas = np.linspace(0, 1, n_steps)

    fig = plt.figure(figsize=(n_steps * 1.5, 7))
    gs  = fig.add_gridspec(3, n_steps, height_ratios=[1, 1, 1.6], hspace=0.4)

    for col, alpha in enumerate(alphas):
        # AE interpolation
        z_ae  = (1 - alpha) * latent_ae[idx_a]  + alpha * latent_ae[idx_b]
        gen_ae = (ae_model.decode(z_ae.reshape(1, -1)).flatten() > 0.5).astype(float)
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(gen_ae.reshape(shape), cmap="binary", vmin=0, vmax=1)
        ax.set_title(f"{alpha:.2f}", fontsize=7)
        ax.axis("off")

        # VAE interpolation
        z_vae = (1 - alpha) * latent_vae[idx_a] + alpha * latent_vae[idx_b]
        gen_vae = (vae_model.decode(z_vae.reshape(1, -1)).flatten() > 0.5).astype(float)
        ax2 = fig.add_subplot(gs[1, col])
        ax2.imshow(gen_vae.reshape(shape), cmap="binary", vmin=0, vmax=1)
        ax2.axis("off")

    # Row labels
    fig.text(0.01, 0.78, "AE",  va="center", fontsize=11, fontweight="bold", color="steelblue")
    fig.text(0.01, 0.55, "VAE", va="center", fontsize=11, fontweight="bold", color="darkorange")

    # Mini mapa del espacio latente con trayectoria
    ax_map_ae  = fig.add_subplot(gs[2, :n_steps//2])
    ax_map_vae = fig.add_subplot(gs[2, n_steps//2:])

    for ax_map, latent, model_name, color in [
        (ax_map_ae,  latent_ae,  "AE",  "steelblue"),
        (ax_map_vae, latent_vae, "VAE", "darkorange"),
    ]:
        ax_map.scatter(latent[:, 0], latent[:, 1], c="lightgray", s=30, zorder=2)
        path = np.array([(1 - a) * latent[idx_a] + a * latent[idx_b] for a in alphas])
        ax_map.plot(path[:, 0], path[:, 1], "-o", color=color, linewidth=2,
                    markersize=5, zorder=3)
        ax_map.scatter(*latent[idx_a], c="green",  s=80, zorder=4, label="feliz")
        ax_map.scatter(*latent[idx_b], c="red",    s=80, zorder=4, label="triste")
        ax_map.set_title(f"Trayectoria {model_name}", fontsize=9)
        ax_map.set_xlabel("z₁"); ax_map.set_ylabel("z₂")
        ax_map.legend(fontsize=7); ax_map.grid(True, alpha=0.3)

    fig.suptitle("Interpolación feliz → triste: AE vs VAE", fontsize=13)
    if filename:
        _save(filename)


# ---------------------------------------------------------------------------
# Visualizaciones para Olivetti Faces (imágenes continuas)
# ---------------------------------------------------------------------------

def _pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA con numpy. Retorna (proyección N×2, varianza explicada [2])."""
    X_c = X - X.mean(axis=0)
    cov = (X_c.T @ X_c) / len(X)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    projected = X_c @ vecs[:, order[:2]]
    ratio = vals[order[:2]] / vals.sum()
    return projected, ratio


def plot_face_reconstructions(
    data: np.ndarray,
    reconstructed: np.ndarray,
    size: int = 32,
    n_show: int = 8,
    title: str | None = None,
    filename: str | None = None,
) -> None:
    """3 filas: original | reconstruido | |diferencia| para n_show caras."""
    indices = np.linspace(0, len(data) - 1, n_show, dtype=int)
    shape = (size, size)
    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 1.6, 5.5))
    row_labels = ["Original", "Reconstruido", "|Diferencia|"]
    cmaps = ["gray", "gray", "hot"]

    for col, idx in enumerate(indices):
        orig  = data[idx].reshape(shape)
        recon = reconstructed[idx].reshape(shape)
        imgs  = [orig, recon, np.abs(orig - recon)]
        for row, (img, cmap) in enumerate(zip(imgs, cmaps)):
            axes[row, col].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[row, col].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0, labelpad=60, va="center")

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_olivetti_training(
    results: dict,
    title: str,
    filename: str | None = None,
) -> None:
    """Curvas de loss + barra de loss final para múltiples configuraciones."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = plt.cm.tab10.colors
    names = list(results.keys())

    for i, (name, r) in enumerate(results.items()):
        h = r["history"]
        c = colors[i % len(colors)]
        ax1.semilogy(h["loss"], label=name, color=c, alpha=0.85)
        if h["converged_at"]:
            ax1.axvline(h["converged_at"], color=c, linestyle=":", alpha=0.5)

    ax1.set(title="Pérdida (log)", xlabel="Época")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    final_losses = [r["history"]["loss"][-1] for r in results.values()]
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]
    ax2.bar(range(len(names)), final_losses, color=bar_colors, alpha=0.85)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Loss final")
    ax2.set_title("Comparación loss final")
    ax2.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(final_losses):
        ax2.text(i, v * 1.01, f"{v:.5f}", ha="center", fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if filename:
        _save(filename)

    print(f"\n{'Config':<22} {'Loss final':<15} {'Épocas'}")
    print("-" * 55)
    for name, r in results.items():
        h = r["history"]
        epochs = h["converged_at"] or len(h["loss"])
        print(f"{name:<22} {h['loss'][-1]:<15.6f} {epochs}")


def plot_latent_faces_2d(
    latent: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    size: int = 32,
    filename: str | None = None,
) -> None:
    """Espacio latente 2D con thumbnail de cada cara (borde coloreado por persona)."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    shape = (size, size)
    margin = 0.5
    fig, ax = plt.subplots(figsize=(16, 14))

    for i, (z, face) in enumerate(zip(latent, data)):
        imgbox = OffsetImage(face.reshape(shape), zoom=0.85, cmap="gray")
        imgbox.image.axes = ax
        color = plt.cm.tab20(labels[i] / 40.0)
        ab = AnnotationBbox(
            imgbox, (z[0], z[1]),
            frameon=True,
            bboxprops=dict(edgecolor=color, linewidth=1.2, boxstyle="round,pad=0.05"),
            pad=0.05,
        )
        ax.add_artist(ab)

    ax.set_xlim(latent[:, 0].min() - margin, latent[:, 0].max() + margin)
    ax.set_ylim(latent[:, 1].min() - margin, latent[:, 1].max() + margin)
    ax.set_title("Espacio latente 2D — Olivetti (color = persona)", fontsize=13)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.grid(True, alpha=0.2, zorder=0)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_latent_pca(
    latent: np.ndarray,
    labels: np.ndarray,
    title: str = "Espacio latente — PCA 2D",
    filename: str | None = None,
) -> None:
    """Proyección PCA del espacio latente coloreada por ID de persona."""
    projected, ratio = _pca_2d(latent)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(projected[:, 0], projected[:, 1],
                    c=labels, cmap="tab20", s=55, alpha=0.85, zorder=3)
    plt.colorbar(sc, ax=ax, label="Persona", ticks=range(0, 40, 5))
    ax.set_title(
        f"{title}\nPC1={ratio[0]*100:.1f}%  PC2={ratio[1]*100:.1f}% varianza explicada",
        fontsize=11,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_olivetti_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    size: int = 32,
    n_per_person: int = 3,
    filename: str | None = None,
) -> None:
    """Muestra n_per_person fotos de cada persona (40 personas)."""
    n_persons = 40
    shape = (size, size)
    fig, axes = plt.subplots(n_persons, n_per_person,
                             figsize=(n_per_person * 1.2, n_persons * 1.2))
    for person in range(n_persons):
        idxs = np.where(labels == person)[0][:n_per_person]
        for col, idx in enumerate(idxs):
            axes[person, col].imshow(data[idx].reshape(shape), cmap="gray", vmin=0, vmax=1)
            axes[person, col].axis("off")
        axes[person, 0].set_ylabel(f"P{person}", fontsize=6,
                                    rotation=0, labelpad=20, va="center")
    fig.suptitle(f"Olivetti Faces — {n_persons} personas × {n_per_person} fotos", fontsize=12)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_ae_vs_vae(
    ae_model,
    vae_model,
    latent_ae: np.ndarray,
    latent_vae: np.ndarray,
    n_samples: int = 16,
    shape: tuple[int, int] = (9, 9),
    filename: str | None = None,
) -> None:
    """
    Compara AE vs VAE sampleando puntos aleatorios en el espacio latente.
    Muestra que el VAE produce caras coherentes fuera de los puntos de entrenamiento,
    mientras el AE produce artefactos.
    """
    rng = np.random.default_rng(1)

    # Para AE: samplear dentro del rango del espacio latente aprendido
    z1 = rng.uniform(latent_ae[:, 0].min(), latent_ae[:, 0].max(), n_samples)
    z2 = rng.uniform(latent_ae[:, 1].min(), latent_ae[:, 1].max(), n_samples)
    z_ae = np.stack([z1, z2], axis=1)

    # Para VAE: samplear del prior N(0,1)
    z_vae = rng.standard_normal((n_samples, 2))

    gen_ae  = ae_model.decode(z_ae)
    gen_vae = vae_model.decode(z_vae)

    cols = 8
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 1.4, rows * 3.5))

    for i in range(n_samples):
        r_ae  = (i // cols) * 2
        r_vae = r_ae + 1
        col   = i % cols
        axes[r_ae,  col].imshow((gen_ae[i]  > 0.5).reshape(shape), cmap="binary", vmin=0, vmax=1)
        axes[r_vae, col].imshow((gen_vae[i] > 0.5).reshape(shape), cmap="binary", vmin=0, vmax=1)
        axes[r_ae,  col].axis("off")
        axes[r_vae, col].axis("off")

    for r in range(0, rows * 2, 2):
        axes[r,     0].set_ylabel("AE",  fontsize=9, rotation=0, labelpad=28, va="center")
        axes[r + 1, 0].set_ylabel("VAE", fontsize=9, rotation=0, labelpad=28, va="center")

    fig.suptitle("AE vs VAE — muestreo aleatorio del espacio latente", fontsize=12)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_vae_training_continuous(
    history: dict,
    filename: str | None = None,
) -> None:
    """Curvas de entrenamiento para VAE sobre datos continuos (sin métrica de px error)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.semilogy(history["loss"],  label="Total",          color="black",      alpha=0.9)
    ax1.semilogy(history["recon"], label="Reconstrucción", color="steelblue",  alpha=0.85)
    if history.get("converged_at"):
        ax1.axvline(history["converged_at"], color="gray", linestyle=":", alpha=0.7,
                    label=f"Early stop (época {history['converged_at']})")
    ax1.set(title="Pérdida total y reconstrucción (log)", xlabel="Época")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["kl"], color="darkorange", alpha=0.85)
    if history.get("converged_at"):
        ax2.axvline(history["converged_at"], color="gray", linestyle=":", alpha=0.7)
    ax2.set(title="Divergencia KL — verificar que no colapse a 0", xlabel="Época")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Entrenamiento VAE — Olivetti Faces", fontsize=13)
    plt.tight_layout()
    if filename:
        _save(filename)


def plot_ae_vs_vae_olivetti(
    ae_model,
    vae_model,
    data: np.ndarray,
    labels: np.ndarray,
    latent_ae: np.ndarray,
    latent_vae: np.ndarray,
    size: int = 32,
    n_samples: int = 8,
    filename: str | None = None,
) -> None:
    """
    Compara AE vs VAE en Olivetti:
    - Fila superior: PCA del espacio latente de cada modelo (coloreado por persona)
    - Fila inferior: muestras aleatorias decodificadas (AE desde su rango, VAE desde N(0,I))
    """
    rng = np.random.default_rng(1)
    shape = (size, size)

    # Proyección PCA 2D de ambos espacios latentes
    def pca2d(z):
        z_c = z - z.mean(axis=0)
        cov = z_c.T @ z_c / (len(z_c) - 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        return z_c @ vecs[:, order[:2]], vals[order[:2]] / vals.sum()

    proj_ae,  ratio_ae  = pca2d(latent_ae)
    proj_vae, ratio_vae = pca2d(latent_vae)

    # Muestras: AE desde bounding box del latente, VAE desde prior N(0,I)
    z_ae = np.stack([
        rng.uniform(latent_ae[:, d].min(), latent_ae[:, d].max(), n_samples)
        for d in range(latent_ae.shape[1])
    ], axis=1)
    z_vae = rng.standard_normal((n_samples, vae_model.latent_dim))
    gen_ae  = ae_model.decode(z_ae)
    gen_vae = vae_model.decode(z_vae)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=0.35, wspace=0.3)

    # PCA — AE
    ax_ae = fig.add_subplot(gs[0, 0])
    sc = ax_ae.scatter(proj_ae[:, 0], proj_ae[:, 1],
                       c=labels, cmap="tab20", s=40, alpha=0.8)
    ax_ae.set_title(
        f"Espacio latente AE — PCA\nPC1={ratio_ae[0]*100:.1f}%  PC2={ratio_ae[1]*100:.1f}%",
        fontsize=10)
    ax_ae.set_xlabel("PC1"); ax_ae.set_ylabel("PC2")
    ax_ae.grid(True, alpha=0.3)

    # PCA — VAE
    ax_vae = fig.add_subplot(gs[0, 1])
    ax_vae.scatter(proj_vae[:, 0], proj_vae[:, 1],
                   c=labels, cmap="tab20", s=40, alpha=0.8)
    ax_vae.set_title(
        f"Espacio latente VAE — PCA\nPC1={ratio_vae[0]*100:.1f}%  PC2={ratio_vae[1]*100:.1f}%",
        fontsize=10)
    ax_vae.set_xlabel("PC1"); ax_vae.set_ylabel("PC2")
    ax_vae.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax_vae, label="Persona", ticks=range(0, 40, 10))

    # Muestras — fila inferior
    gs_bot = gs[1, :].subgridspec(2, n_samples, hspace=0.05, wspace=0.05)
    for i in range(n_samples):
        for row, (gen, label) in enumerate([(gen_ae, "AE"), (gen_vae, "VAE")]):
            ax = fig.add_subplot(gs_bot[row, i])
            ax.imshow(gen[i].reshape(shape), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=28, va="center")

    fig.suptitle("AE vs VAE — estructura latente y calidad generativa (Olivetti)", fontsize=12)
    if filename:
        _save(filename)
