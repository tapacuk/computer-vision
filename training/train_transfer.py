import torch
import torch.nn as nn
from tqdm import tqdm
import os


def train_transfer(model, train_loader, val_loader, device,
                   epochs=10,
                   lr_backbone=1e-4,
                   lr_head=1e-3,
                   weight_decay=1e-5,
                   patience=5,
                   save_path="checkpoints/transfer_best.pth"):

    # поділ параметрів моделі
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "fc" in name:   # останній шар
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    no_improve = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # train
        model.train()
        train_loss = 0
        total, correct = 0, 0

        for x, y in tqdm(train_loader, desc="Training"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0
        total, correct = 0, 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation"):
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        val_loss /= len(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # early stop
        if val_loss < best_val_loss:
            print("Найкраща модель збережена")
            best_val_loss = val_loss
            no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            print("early stopping")
            break

    return history
