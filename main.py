import numpy as np

from src.data import load_data
from src.autoencoder import build_shallow, build_deep
from src.losses import BCE, MSE
from src.optimizers import Adam, SGD
from src.training import train, pixel_error
from src.visualization import plot_chars, plot_training, plot_latent_space

SEED = 42
MAX_EPOCHS = 50_000


def experiment_architecture(data: np.ndarray) -> dict:
    """Exp 1: Shallow vs Deep, fijo BCE + Adam."""
    print("\n" + "=" * 60)
    print("EXPERIMENTO 1: Arquitectura (BCE + Adam lr=1e-3)")
    print("=" * 60)

    results = {}
    for name, build_fn in [("Shallow (35-16-2)", build_shallow), ("Deep (35-32-16-8-2)", build_deep)]:
        print(f"\n--- {name} ---")
        model = build_fn(seed=SEED)
        results[name] = {
            "model": model,
            "history": train(model, data, Adam(lr=1e-3), BCE(), epochs=MAX_EPOCHS),
        }

    plot_training(results, "Experimento 1: Arquitectura (BCE + Adam lr=1e-3)",
                  filename="exp1_arquitectura.png")
    return results


def experiment_loss(data: np.ndarray, build_fn, arch_name: str) -> dict:
    """Exp 2: BCE vs MSE, fijo mejor arch + Adam."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENTO 2: Loss function ({arch_name} + Adam lr=1e-3)")
    print("=" * 60)

    results = {}
    for name, loss_fn in [("BCE", BCE()), ("MSE", MSE())]:
        print(f"\n--- {name} ---")
        model = build_fn(seed=SEED)
        results[name] = {
            "model": model,
            "history": train(model, data, Adam(lr=1e-3), loss_fn, epochs=MAX_EPOCHS),
        }

    plot_training(results, f"Experimento 2: Loss function ({arch_name} + Adam lr=1e-3)",
                  filename="exp2_loss.png")
    return results


def experiment_optimizer(data: np.ndarray, build_fn, arch_name: str, loss_fn, loss_name: str) -> dict:
    """Exp 3: Adam vs SGD, fijo mejor arch + mejor loss."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENTO 3: Optimizador ({arch_name} + {loss_name})")
    print("=" * 60)

    configs = [
        ("Adam (lr=1e-3)",         Adam(lr=1e-3)),
        ("SGD (lr=0.01, mom=0.9)", SGD(lr=0.01, momentum=0.9)),
    ]

    results = {}
    for name, optimizer in configs:
        print(f"\n--- {name} ---")
        model = build_fn(seed=SEED)
        results[name] = {
            "model": model,
            "history": train(model, data, optimizer, loss_fn, epochs=MAX_EPOCHS),
        }

    plot_training(results, f"Experimento 3: Optimizador ({arch_name} + {loss_name})",
                  filename="exp3_optimizador.png")
    return results


def analyze_best_model(model, data: np.ndarray, labels: list[str]) -> None:
    print("\n" + "=" * 60)
    print("ANÁLISIS DEL MEJOR MODELO")
    print("=" * 60)

    reconstructed = model.forward(data)
    errors = pixel_error(reconstructed, data)

    print("\nError por carácter:")
    for label, err in zip(labels, errors):
        mark = " <-- !" if err > 1 else ""
        print(f"  '{label}': {err:.0f} px{mark}")
    print(f"\nMax error: {errors.max():.0f} px | Mean error: {errors.mean():.2f} px")

    plot_chars(list(data), labels, title="Originales", filename="chars_originales.png")
    plot_chars(list(reconstructed), labels, threshold=0.5,
               title="Reconstruidos (umbral 0.5)", filename="chars_reconstruidos.png")

    latent = model.encode(data)
    plot_latent_space(latent, labels, filename="espacio_latente.png")

    _generate_new_chars(model, data, labels, latent)


def _generate_new_chars(model, data: np.ndarray, labels: list[str], latent: np.ndarray) -> None:
    N = 7
    z1_range = np.linspace(latent[:, 0].min() - 0.5, latent[:, 0].max() + 0.5, N)
    z2_range = np.linspace(latent[:, 1].max() + 0.5, latent[:, 1].min() - 0.5, N)

    grid_chars = []
    for z2 in z2_range:
        for z1 in z1_range:
            z = np.array([[z1, z2]])
            grid_chars.append(model.decode(z).flatten())

    plot_chars(grid_chars, cols=N, threshold=0.5,
               title=f"Generación: muestreo del espacio latente ({N}×{N})",
               filename="generacion_grilla.png")

    idx_a, idx_z = labels.index("a"), labels.index("z")
    alphas = np.linspace(0, 1, 9)
    interp_chars = [
        model.decode(((1 - a) * latent[idx_a] + a * latent[idx_z]).reshape(1, -1)).flatten()
        for a in alphas
    ]
    interp_labels = [f"{a:.2f}" for a in alphas]

    plot_chars(interp_chars, interp_labels, cols=9, threshold=0.5,
               title="Interpolación en espacio latente: 'a' → 'z'",
               filename="generacion_interpolacion.png")


def _best_key(results: dict) -> str:
    return min(results, key=lambda k: results[k]["history"]["converged_at"] or MAX_EPOCHS + 1)


def main() -> None:
    data, labels = load_data()
    print(f"Dataset cargado: {data.shape[0]} muestras de {data.shape[1]} features")

    exp1 = experiment_architecture(data)
    best_arch_name = _best_key(exp1)
    best_build_fn = build_shallow if "Shallow" in best_arch_name else build_deep
    print(f"\nMejor arquitectura: {best_arch_name}")

    exp2 = experiment_loss(data, best_build_fn, best_arch_name)
    best_loss_name = _best_key(exp2)
    best_loss_fn = BCE() if best_loss_name == "BCE" else MSE()
    print(f"\nMejor loss: {best_loss_name}")

    exp3 = experiment_optimizer(data, best_build_fn, best_arch_name, best_loss_fn, best_loss_name)
    best_opt_name = _best_key(exp3)
    print(f"\nMejor optimizador: {best_opt_name}")

    best_model = exp3[best_opt_name]["model"]
    print(f"\nConfiguracion optima: {best_arch_name} + {best_loss_name} + {best_opt_name}")

    analyze_best_model(best_model, data, labels)


if __name__ == "__main__":
    main()
