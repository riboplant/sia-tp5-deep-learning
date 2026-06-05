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
) -> None:
    imgs = [(c > threshold).astype(float) if threshold is not None else c for c in chars]
    n = len(imgs)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.6))
    axes = np.array(axes).flatten()
    for i, img in enumerate(imgs):
        axes[i].imshow(img.reshape(7, 5), cmap="binary", vmin=0, vmax=1)
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
