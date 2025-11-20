# train.py
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataloaders(data_dir, img_size=224, batch_size=32):
    mean = [0.485, 0.456, 0.406]  # Imagenet
    std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

def build_model(num_classes, arch="efficientnet_b0"):
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        # Por defecto EfficientNet_B0: muy buena relación precisión/velocidad
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

def train(data_dir, out_dir, epochs=10, batch_size=32, lr=1e-3, arch="efficientnet_b0", img_size=224):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = get_dataloaders(data_dir, img_size, batch_size)
    model = build_model(len(classes), arch=arch).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_path = out_dir / "best_state_dict.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total, correct = 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss: {train_loss:.4f} acc: {train_acc:.3f} | "
              f"val_loss: {val_loss:.4f} acc: {val_acc:.3f}")

        # Guarda el mejor por accuracy de validación
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    # Cargar mejor estado y exportar a TorchScript (para servir estable)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    # Ejemplo de entrada para trazar el modelo
    example = torch.randn(1, 3, img_size, img_size).to(DEVICE)
    traced = torch.jit.trace(model, example)
    traced.save(str(out_dir / "model.pt"))

    # Guardar labels
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    # Log
    with open(out_dir / "train_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}Z "
                f"arch={arch} epochs={epochs} bs={batch_size} img={img_size} "
                f"best_val_acc={best_val_acc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Carpeta con train/ y val/")
    parser.add_argument("--out_dir",  default="models", help="Dónde guardar model.pt y labels.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arch", choices=["efficientnet_b0", "resnet18"], default="efficientnet_b0")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    train(**vars(args))
