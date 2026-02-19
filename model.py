import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import argparse
from aug_utils import transforms
from torch.amp import autocast, GradScaler
import pandas as pd
from dataset import ImageDataset

# ── Model ─────────────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

scaler = GradScaler()


def build_model(cloud) -> nn.Module:
    model = timm.create_model("convnext_large_in22k", pretrained=True)
    model.head.fc = nn.Linear(in_features=model.head.fc.in_features, out_features=8)

    # noinspection PyArgumentList
    model = model.to(device='cuda', memory_format=torch.channels_last)

    if cloud:
        model = torch.compile(model, mode='max-autotune')

    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        epoch: int,
) -> float:
    model.train()
    running_loss = 0.0

    loop = tqdm(loader, desc=f"Epoch {epoch:>3}/{EPOCHS}", leave=True)

    for images, labels in loop:
        images = images.to(DEVICE, memory_format=torch.channels_last)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(images)
            loss = criterion(predictions, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()

    return running_loss / len(loader)


def train(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        early_stopping_delta: float = 1e-6
) -> None:
    best_loss = float('inf')
    prev_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = train_one_epoch(
            model, loader, criterion, optimizer, scheduler, epoch
        )
        print(f"  └─ avg loss: {epoch_loss:.4f}\n")
        if abs(prev_loss - epoch_loss) < early_stopping_delta:
            break
        prev_loss = epoch_loss
        if epoch_loss < best_loss:
            torch.save(model.state_dict(), f'./models/convnext_large.pth')
            print("saved model", epoch_loss)
            best_loss = epoch_loss


# ── Entry point ───────────────────────────────────────────────────────────────

def main(data_dir: str, img_size: int, delta_es: float, cloud) -> None:
    dataset = ImageDataset(data_dir, img_size)
    if cloud is True:
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,  # Use 80% of your 20 vCPUs
            pin_memory=True,  # Critical for fast CPU→GPU transfer
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,  # Prefetch 4 batches per worker = 64 batches ahead
            multiprocessing_context='fork',  # Faster than spawn on Linux
            drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=BATCH_SIZE
        )

    model = build_model(cloud)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # optimizer params
    # different lr for head because newly initialised
    backbone_params = [p for n, p in model.named_parameters()
                       if not n.startswith("head")]
    head_params = [p for n, p in model.named_parameters()
                   if n.startswith("head")]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    train(model, loader, criterion, optimizer, scheduler, delta_es)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--epochs', type=int, default=50)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--img_size', type=int, default=512)
    args.add_argument('--data_dir', type=str, default="./data")
    args.add_argument('--delta_es', default=1e-4, type=float)
    args.add_argument('--local', action='store_true', default=False)

    args = args.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────────

    DEVICE = "cuda:0"
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    NUM_CLASSES = 8
    LR = args.lr
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    cloud = True
    data_dir = args.data_dir

    if args.local:
        cloud = False

    df_features = pd.read_csv(f"{data_dir}/train_features.csv").set_index('id')
    df_labels = pd.read_csv(f"{data_dir}/train_labels.csv").set_index("id")

    df = pd.concat([df_features, df_labels], axis=1)

    main(args.data_dir, args.img_size, args.delta_es, cloud)
