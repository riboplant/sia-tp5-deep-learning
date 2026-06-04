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
