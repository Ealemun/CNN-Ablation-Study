import time

import torch
import torch.nn as nn
from tqdm import tqdm


def evaluate(model, loader, device, criterion):
    """
    Évalue le modèle sur un dataset.

    Retourne :
        loss moyenne par échantillon
        accuracy
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            batch_size = x.size(0)

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += batch_size

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=10,
    lr=1e-3,
    verbose=True
):
    """
    Entraîne un modèle et retourne son historique.

    Parameters
    ----------
    model : nn.Module
        Modèle à entraîner.

    train_loader : DataLoader
        DataLoader du dataset d'entraînement.

    val_loader : DataLoader
        DataLoader du dataset de validation.

    device : torch.device
        CPU ou GPU.

    epochs : int
        Nombre d'époques.

    lr : float
        Learning rate utilisé par Adam.

    verbose : bool
        Affiche les métriques à chaque époque.
    """

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_s": [],
    }

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            disable=not verbose
        )

        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)

            # 1. Remet les gradients à zéro
            optimizer.zero_grad()

            # 2. Forward pass
            logits = model(x)

            # 3. Calcul de la loss
            loss = criterion(logits, y)

            # 4. Backpropagation
            loss.backward()

            # 5. Mise à jour des paramètres
            optimizer.step()

            # Statistiques
            batch_size = x.size(0)

            running_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += batch_size

            if verbose:
                progress_bar.set_postfix(
                    loss=loss.item(),
                    acc=correct / total
                )

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
            criterion
        )

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_s"].append(epoch_time)

        if verbose:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"time={epoch_time:.2f}s"
            )

    return history
