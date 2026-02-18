import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from dataset import ImageDataset


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = "cuda:0"
BATCH_SIZE  = 1024
EPOCHS      = 50
NUM_CLASSES = 8
LR          = 6e-2
MOMENTUM    = 0.9
WEIGHT_DECAY= 1e-4


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    for name, param in model.named_parameters():
        if 'layer4' not in name:
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  optim.Optimizer,
    scheduler:  optim.lr_scheduler.LRScheduler,
    epoch:      int,
) -> float:
    model.train()
    running_loss = 0.0

    loop = tqdm(loader, desc=f"Epoch {epoch:>3}/{EPOCHS}", leave=True)

    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()

    return running_loss / len(loader)


def train(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  optim.Optimizer,
    scheduler:  optim.lr_scheduler.LRScheduler,
) -> None:
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = train_one_epoch(
            model, loader, criterion, optimizer, scheduler, epoch
        )
        print(f"  └─ avg loss: {epoch_loss:.4f}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    dataset    = ImageDataset()
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model      = build_model(NUM_CLASSES)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer  = optim.SGD([
        {"params": model.layer4.parameters(), "lr": 1e-3},
        {"params": model.fc.parameters(), "lr": LR}
    ],
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS, eta_min=1e-5
                )

    train(model, loader, criterion, optimizer, scheduler)


if __name__ == "__main__":
    main()