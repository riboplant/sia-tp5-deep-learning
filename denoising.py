import numpy as np

from src.data import load_data
from src.autoencoder import build_deep, build_deep_wide
from src.losses import BCE
from src.optimizers import Adam
from src.noise import salt_and_pepper, masking
from src.training import train_denoising, pixel_error
from src.visualization import (
    plot_noise_levels,
    plot_denoising_summary,
    plot_denoising_comparison,
    plot_denoising_reconstructions,
    plot_latent_space,
)

SEED = 42
MAX_EPOCHS = 50_000
NOISE_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.50]


def run_noise_type(
    name: str,
    noise_fn,
    data: np.ndarray,
    arch_builder,
    arch_label: str,
) -> dict[float, dict]:
    print(f"\n{'=' * 60}")
    print(f"Ruido: {name} | Arquitectura: {arch_label}")
    print("=" * 60)

    results = {}
    for p in NOISE_LEVELS:
        print(f"\n  --- p={p} ---")
        model = arch_builder(seed=SEED)
        optimizer = Adam(lr=1e-3)
        loss_fn = BCE()
        noise_fn_p = lambda d, rng, _p=p: noise_fn(d, _p, rng)
        history = train_denoising(
            model, data, optimizer, loss_fn, noise_fn_p,
            epochs=MAX_EPOCHS, seed=SEED,
        )
        results[p] = {"model": model, "history": history}

    return results


def needs_wide(results: dict[float, dict]) -> bool:
    """Retorna True si algún nivel no convergió."""
    return any(r["history"]["converged_at"] is None for r in results.values())


def main() -> None:
    data, labels = load_data()
    print(f"Dataset: {data.shape[0]} muestras de {data.shape[1]} features")

    # ------------------------------------------------------------------ #
    # Visualizar niveles de ruido antes de entrenar                        #
    # ------------------------------------------------------------------ #
    plot_noise_levels(data, salt_and_pepper, NOISE_LEVELS, labels,
                      "Salt & Pepper", filename="dn_noise_sp.png")
    plot_noise_levels(data, masking, NOISE_LEVELS, labels,
                      "Masking", filename="dn_noise_masking.png")

    # ------------------------------------------------------------------ #
    # Experimento: Salt & Pepper con arquitectura Deep                     #
    # ------------------------------------------------------------------ #
    sp_results = run_noise_type("Salt & Pepper", salt_and_pepper, data,
                                build_deep, "Deep (35-32-16-8-2)")
    plot_denoising_summary(sp_results, "Salt & Pepper", "Deep (35-32-16-8-2)",
                           filename="dn_sp_deep_training.png")

    # Si no converge con Deep, reintentar con Deep Wide
    if needs_wide(sp_results):
        print("\n  [!] Algunos niveles no convergieron con Deep — reintentando con Deep Wide")
        sp_results_wide = run_noise_type("Salt & Pepper", salt_and_pepper, data,
                                         build_deep_wide, "Deep Wide (35-64-32-16-2)")
        plot_denoising_summary(sp_results_wide, "Salt & Pepper", "Deep Wide (35-64-32-16-2)",
                               filename="dn_sp_wide_training.png")
        sp_final = sp_results_wide
        sp_arch_label = "Deep Wide"
        sp_builder = build_deep_wide
    else:
        sp_final = sp_results
        sp_arch_label = "Deep"
        sp_builder = build_deep

    # ------------------------------------------------------------------ #
    # Experimento: Masking con la misma arquitectura elegida               #
    # ------------------------------------------------------------------ #
    mask_results = run_noise_type("Masking", masking, data,
                                  sp_builder, f"{sp_arch_label}")
    plot_denoising_summary(mask_results, "Masking", sp_arch_label,
                           filename="dn_masking_training.png")

    if needs_wide(mask_results) and sp_builder is not build_deep_wide:
        print("\n  [!] Masking tampoco convergió — reintentando con Deep Wide")
        mask_results_wide = run_noise_type("Masking", masking, data,
                                           build_deep_wide, "Deep Wide (35-64-32-16-2)")
        plot_denoising_summary(mask_results_wide, "Masking", "Deep Wide (35-64-32-16-2)",
                               filename="dn_masking_wide_training.png")
        mask_final = mask_results_wide
    else:
        mask_final = mask_results

    # ------------------------------------------------------------------ #
    # Comparación directa: Salt & Pepper vs Masking                        #
    # ------------------------------------------------------------------ #
    plot_denoising_comparison(sp_final, mask_final, filename="dn_comparison.png")

    # ------------------------------------------------------------------ #
    # Visualización de reconstrucciones por nivel de ruido                 #
    # ------------------------------------------------------------------ #
    # Usar el modelo entrenado al nivel más alto que convergió
    best_sp_p = max(
        (p for p, r in sp_final.items() if r["history"]["converged_at"] is not None),
        default=NOISE_LEVELS[0],
    )
    best_mask_p = max(
        (p for p, r in mask_final.items() if r["history"]["converged_at"] is not None),
        default=NOISE_LEVELS[0],
    )

    print(f"\nMejor modelo Salt & Pepper: p={best_sp_p}")
    plot_denoising_reconstructions(
        sp_final[best_sp_p]["model"], data, salt_and_pepper,
        NOISE_LEVELS, labels, "Salt & Pepper",
        filename="dn_sp_reconstructions.png",
    )

    print(f"Mejor modelo Masking: p={best_mask_p}")
    plot_denoising_reconstructions(
        mask_final[best_mask_p]["model"], data, masking,
        NOISE_LEVELS, labels, "Masking",
        filename="dn_masking_reconstructions.png",
    )

    # ------------------------------------------------------------------ #
    # Espacio latente del modelo entrenado con mayor ruido                 #
    # ------------------------------------------------------------------ #
    print("\nEspacio latente — modelo Salt & Pepper con mayor nivel de ruido convergido")
    best_sp_model = sp_final[best_sp_p]["model"]
    latent = best_sp_model.encode(data)
    plot_latent_space(latent, labels, filename=f"dn_latent_sp_p{best_sp_p}.png")


if __name__ == "__main__":
    main()
