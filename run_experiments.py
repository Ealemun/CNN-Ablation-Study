import argparse
import json
import os
import time

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import AblationCNN, build_model, count_params
from train import train_model, evaluate


def get_dataloaders(batch_size=128, data_dir="./data"):
    """
    Construit les DataLoaders train/validation/test pour Fashion-MNIST.
    
        Fashion-MNIST contient :
            - 60 000 images d'entraînement
            - 10 000 images de test
    
        On divise les 60 000 images d'entraînement en :
            - 50 000 images pour le train
            - 10 000 images pour la validation
    
    Le test officiel de 10 000 images reste complètement séparé.

    Returns
    -------
    train_loader : DataLoader
        Loader du dataset d'entraînement.

    val_loader : DataLoader
        Loader du dataset de validation.

    test_loader : DataLoader
        Loader du dataset de test.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.2860,),
            (0.3530,)
        ),
    ])

    # Dataset d'entraînement officiel : 60 000 images
    full_train_set = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Dataset de test officiel : 10 000 images
    test_set = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Division du train officiel :
    # 50 000 train / 10 000 validation
    train_set, val_set = torch.utils.data.random_split(
        full_train_set,
        [50000, 10000],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader



def main():
    parser = argparse.ArgumentParser(
        description="Run CNN ablation study on Fashion-MNIST."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training and validation batch size."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate."
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=32,
        help="Number of channels in the CNN."
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=6,
        help="Number of blocks."
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory where results are saved."
    )

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size
    )


    summary = {}

    for variant in AblationCNN.VARIANTS:
        print("\n" + "=" * 60)
        print(f"Training variant: {variant}")
        print("=" * 60)

        model = build_model(
            variant=variant,
            channels=args.channels,
            num_blocks=args.num_blocks,
        )

        params = count_params(model)

        print(f"Trainable parameters: {params:,}")

        start_time = time.time()

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            verbose=True,
        )

        total_time = time.time() - start_time

        test_criterion = torch.nn.CrossEntropyLoss()

        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            criterion=test_criterion,
        )


        history_path = os.path.join(
            args.results_dir,
            f"history_{variant}.json"
        )

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        best_val_acc = max(history["val_acc"])
        final_val_acc = history["val_acc"][-1]

        summary[variant] = {
            "params": params,
            "best_val_acc": best_val_acc,
            "final_val_acc": final_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "total_time_s": total_time,
        }


        print(f"Best validation accuracy:  {best_val_acc:.4f}")
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        print(f"Total training time:       {total_time:.2f}s")
        print(f"Test loss:                 {test_loss:.4f}")
        print(f"Test accuracy:             {test_acc:.4f}")


    summary_path = os.path.join(
        args.results_dir,
        "summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"{'Variant':<22}"
        f"{'Params':>12}"
        f"{'Best Val Acc':>16}"
        f"{'Final Val Acc':>16}"
        f"{'Test Acc':>12}"
        f"{'Time (s)':>12}"
    )


    print("-" * 80)

    for variant, result in summary.items():
        print(
            f"{variant:<22}"
            f"{result['params']:>12,}"
            f"{result['best_val_acc']:>16.4f}"
            f"{result['final_val_acc']:>16.4f}"
            f"{result['test_acc']:>12.4f}"
            f"{result['total_time_s']:>12.2f}"
        )



if __name__ == "__main__":
    main()
