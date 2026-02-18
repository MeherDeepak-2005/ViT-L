import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import vit_l_16
from torchvision.models import ViT_L_16_Weights
from tqdm import tqdm
import argparse

from dataset import ImageDataset

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(img_size) -> nn.Module:
    model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
    model.image_size = img_size

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "encoder_layer_10" in name:
            param.requires_grad = True
        if "encoder_layer_11" in name:
            param.requires_grad = True
        if "heads in name":
            param.requires_grad = True
    model.heads.head = nn.Linear(768, 8, bias=True)
    model = model.to(device='cuda')
    model = model.to(memory_format=torch.channels_last)

    model = torch.compile(model, mode='max-autotune')

    return model

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

def main(data_dir: str, img_size: int) -> None:
    dataset    = ImageDataset(data_dir, img_size)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # Use 80% of your 20 vCPUs
        pin_memory=True,  # Critical for fast CPU→GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch 4 batches per worker = 64 batches ahead
        multiprocessing_context='fork',  # Faster than spawn on Linux
    )

    model      = build_model(img_size=img_size)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer  = optim.AdamW([
        {"params": model.encoder.layers.encoder_layer_10.parameters(), "lr": LR},
        {"params": model.encoder.layers.encoder_layer_11.parameters(), "lr": LR},
        {"params": model.heads.parameters(), "lr": 1e-2}
    ],
        weight_decay=0.05
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS, eta_min=1e-5
                )

    train(model, loader, criterion, optimizer, scheduler)
    model.save("./models/ViT_Base")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=4096)
    args.add_argument('--epochs', type=int, default=50)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--img_size',type=int, default=512)
    args.add_argument('--data_dir', type=str)

    args = args.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────────

    DEVICE = "cuda:0"
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    NUM_CLASSES = 8
    LR = args.lr
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4



    main(args.data_dir, args.img_size)