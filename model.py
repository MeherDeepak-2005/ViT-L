import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from aug_utils import gridmask, mix_batch, soft_cross_entropy
from torch.amp import autocast, GradScaler
from dataset import ImageDataset

# ── Perf flags ────────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(cloud: bool) -> nn.Module:
    model = timm.create_model("convnext_large_in22k", pretrained=True)
    model.head.fc = nn.Linear(model.head.fc.in_features, out_features=8)
    model = model.to(device='cuda', memory_format=torch.channels_last)
    if cloud:
        model = torch.compile(model, mode='max-autotune')
    return model


# ── One epoch ─────────────────────────────────────────────────────────────────

def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        scaler: GradScaler,
        epoch: int,
        epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch:>3}/{epochs}", leave=True)

    for images, labels in loop:
        images = images.to(DEVICE, memory_format=torch.channels_last)
        labels = labels.to(DEVICE)

        images = gridmask(images)
        images, soft_labels = mix_batch(images, labels, alpha=0.4)

        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(images)
            loss = soft_cross_entropy(logits, soft_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE, memory_format=torch.channels_last)
        labels = labels.to(DEVICE)
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(images)
            # val uses hard integer labels → standard cross entropy
            loss = nn.functional.cross_entropy(logits, labels)
        running_loss += loss.item()
    return running_loss / len(loader)


# ── Full training loop ────────────────────────────────────────────────────────

def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        early_stopping_patience: int = 5,
        save_path: str = './models/convnext_large.pth',
) -> None:
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch, EPOCHS)
        val_loss = validate(model, val_loader)

        print(f"  └─ train loss: {train_loss:.4f}  |  val loss: {val_loss:.4f}")

        scheduler.step()

        # Save on best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✔ saved model  (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_loader(dataset: ImageDataset, labels_int: np.ndarray,
                cloud: bool, train: bool = True) -> DataLoader:
    """
    train=True  → WeightedRandomSampler for class balance
    train=False → sequential, no shuffle
    """
    if train:
        class_freqs = np.bincount(labels_int) / len(labels_int)
        class_weights = 1.0 / class_freqs
        sample_weights = class_weights[labels_int]  # (N,) one weight per sample
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # sampler and shuffle are mutually exclusive
    else:
        sampler = None
        shuffle = False

    if cloud:
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            multiprocessing_context='fork',
            drop_last=train,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=train,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(data_dir: str, img_size: int, cloud: bool) -> None:
    # ── Load CSVs ──────────────────────────────────────────────────────────────
    df_features = pd.read_csv(f"{data_dir}/train_features.csv").set_index('id')
    df_labels = pd.read_csv(f"{data_dir}/train_labels.csv").set_index("id")
    df = pd.concat([df_features, df_labels], axis=1)

    # File paths — prepend data_dir so the path is complete
    all_paths = (data_dir + "/" + df['filepath']).tolist()

    # One-hot label columns (all columns except filepath)
    label_cols = df_labels.columns.tolist()
    labels_onehot = df[label_cols].values.astype(np.float32)  # (N, 8)
    labels_int = np.argmax(labels_onehot, axis=1)  # (N,)  integer

    # ── Stratified train / val split ───────────────────────────────────────────
    train_idx, val_idx = train_test_split(
        np.arange(len(all_paths)),
        test_size=0.2,
        stratify=labels_int,  # preserves class balance in both splits
        random_state=42
    )

    train_paths = [all_paths[i] for i in train_idx]
    val_paths = [all_paths[i] for i in val_idx]
    train_labels = labels_onehot[train_idx]  # (N_train, 8)
    val_labels = labels_onehot[val_idx]  # (N_val,   8)

    print(f"Train: {len(train_paths)}  |  Val: {len(val_paths)}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_dataset = ImageDataset(train_paths, train_labels, train=True, img_size=img_size)
    val_dataset = ImageDataset(val_paths, val_labels, train=False, img_size=img_size)

    # ── Loaders ────────────────────────────────────────────────────────────────
    # labels_int[train_idx] for sampler weights
    train_labels_int = labels_int[train_idx]
    train_loader = make_loader(train_dataset, train_labels_int, cloud, train=True)
    val_loader = make_loader(val_dataset, None, cloud, train=False)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(cloud)

    # ── Optimizer — lower lr for pretrained backbone, higher for new head ──────
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    train(model, train_loader, val_loader, optimizer, scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    DEVICE = "cuda:0"
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    cloud = not args.local

    main(args.data_dir, args.img_size, cloud)
