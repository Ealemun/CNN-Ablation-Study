import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt



VARIANTS = (
    "baseline",
    "bottleneck",
    "bottleneck_bn",
    "bottleneck_bn_skip",
)


def load_histories(results_dir):
    """
    Charge les historiques de chaque variante.

    Retourne
    --------
    histories : dict
        Dictionnaire de la forme :
        {
            "baseline": {...},
            "bottleneck": {...},
            ...
        }
    """

    histories = {}

    for variant in VARIANTS:
        path = os.path.join(
            results_dir,
            f"history_{variant}.json"
        )

        with open(path, "r") as f:
            histories[variant] = json.load(f)

    return histories


def plot_metric(histories, metric, ylabel, out_path):
    """
    Génère une courbe comparative pour une métrique.

    Parameters
    ----------
    histories : dict
        Historiques de toutes les variantes.

    metric : str
        Métrique à tracer, par exemple :
        "train_loss", "val_loss", "train_acc", "val_acc".

    ylabel : str
        Label de l'axe Y.

    out_path : str
        Chemin de sauvegarde de la figure.
    """

    plt.figure(figsize=(8, 5))

    for variant, history in histories.items():
        epochs = range(1, len(history[metric]) + 1)

        plt.plot(
            epochs,
            history[metric],
            label=variant
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_test_accuracy(summary, out_path):
    variants = list(summary.keys())
    test_acc = [summary[v]["test_acc"] for v in variants]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(variants, test_acc)

    plt.xlabel("Variant")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy by Variant")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)

    # Affiche la valeur au-dessus de chaque barre
    for bar, acc in zip(bars, test_acc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.4f}",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Plot saved: {out_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Plot CNN ablation study results."
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory containing experiment results."
    )

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # --------------------------------------------------
    # Load histories
    # --------------------------------------------------

    histories = load_histories(args.results_dir)

    # --------------------------------------------------
    # Generate plots
    # --------------------------------------------------

    plot_metric(
        histories,
        metric="train_loss",
        ylabel="Train Loss",
        out_path=os.path.join(
            args.results_dir,
            "train_loss.png"
        ),
    )

    plot_metric(
        histories,
        metric="val_loss",
        ylabel="Validation Loss",
        out_path=os.path.join(
            args.results_dir,
            "val_loss.png"
        ),
    )

    plot_metric(
        histories,
        metric="train_acc",
        ylabel="Train Accuracy",
        out_path=os.path.join(
            args.results_dir,
            "train_acc.png"
        ),
    )

    plot_metric(
        histories,
        metric="val_acc",
        ylabel="Validation Accuracy",
        out_path=os.path.join(
            args.results_dir,
            "val_acc.png"
        ),
    )

    # --------------------------------------------------
    # Load summary
    # --------------------------------------------------

    summary_path = os.path.join(
        args.results_dir,
        "summary.json"
    )

    with open(summary_path, "r") as f:
        summary = json.load(f)

    plot_test_accuracy(
        summary,
        out_path=os.path.join(
            args.results_dir,
            "test_accuracy.png"
        ),
    )


    # --------------------------------------------------
    # Print summary table
    # --------------------------------------------------

    print("\n" + "=" * 90)
    print("EXPERIMENT SUMMARY")
    print("=" * 90)

    print(
        f"{'Variant':<22}"
        f"{'Params':>12}"
        f"{'Best Val Acc':>16}"
        f"{'Test Acc':>12}"
        f"{'Time (s)':>12}"
    )

    print("-" * 90)

    for variant in VARIANTS:
        result = summary[variant]

        print(
            f"{variant:<22}"
            f"{result['params']:>12,}"
            f"{result['best_val_acc']:>16.4f}"
            f"{result['test_acc']:>12.4f}"
            f"{result['total_time_s']:>12.2f}"
        )


if __name__ == "__main__":
    main()
