# train_mvcnn.py
import os
import yaml

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  # optional
import matplotlib.pyplot as plt

from datasets import make_loaders
from models.mvcnn import MVCNN


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train():
    cfg = load_config("configs/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cache_root = cfg["cache_root"]
    img_size = int(cfg["img_size"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])
    num_epochs = int(cfg["num_epochs"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    val_split = float(cfg["val_split"])
    ckpt_path = cfg["checkpoint_path"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # Data
    train_ds, val_ds, test_ds, train_loader, val_loader, _ = make_loaders(
        cache_root, img_size, batch_size, num_workers, val_split=val_split
    )

    num_classes = len(train_ds.dataset.classes)  # random_split wraps original dataset

    model = MVCNN(
        num_classes=num_classes,
        pretrained=True,
        feature_dim=512,
        pooling="max"
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(num_epochs):
        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        for views, labels in train_loader:
            views = views.to(device)   # [B, V, C, H, W]
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(views)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * views.size(0)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = 100.0 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # ---------- VALIDATION ----------
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for views, labels in val_loader:
                views = views.to(device)
                labels = labels.to(device)

                logits, _ = model(views)
                loss = F.cross_entropy(logits, labels)

                val_running_loss += loss.item() * views.size(0)
                preds = logits.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.2f}%"
        )

        # Save best model according to validation accuracy.
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best val acc {best_val_acc:.2f}%, saved to {ckpt_path}")

    # Plot learning curves (optional but handy for the report).
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve - Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train acc")
    plt.plot(epochs, val_accs, label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Learning curve - Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
