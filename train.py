import os

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torch_compile_cache"

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from model_utils import build_model, load_model, get_trainable_params
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from aug_utils import gridmask, mix_batch, soft_cross_entropy
from torch.amp import autocast, GradScaler
from dataset import ImageDataset

# ── Perf flags ────────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


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
            loss = nn.functional.cross_entropy(logits, labels)
        running_loss += loss.item()
    return running_loss / len(loader)


# ── Shared training loop ──────────────────────────────────────────────────────

def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler,
        early_stopping_patience: int,
        save_path: str,
) -> float:
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch, EPOCHS)
        val_loss = validate(model, val_loader)

        print(f"  └─ train loss: {train_loss:.4f}  |  val loss: {val_loss:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✔ saved  (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                print("  Early stopping triggered.")
                break

    return best_val_loss


# ── Phase 1 — head only ───────────────────────────────────────────────────────

def run_phase1(model, train_loader, val_loader, fold, patience) -> str:
    """
    Freeze backbone completely, train WildlifeHead until convergence.
    Saves checkpoint to ./checkpoints/fold{fold}_phase1.pth
    Returns the save path so phase 2 can pick it up automatically.
    """
    save_path = f"./checkpoints/fold{fold}_phase1.pth"
    print(f"\n  [Fold {fold}] Phase 1 — head only (backbone frozen)")

    # get_trainable_params returns only requires_grad=True params — your model_utils fn
    optimizer = optim.AdamW(
        get_trainable_params(model),
        lr=1e-3,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    best_val = train(
        model, train_loader, val_loader,
        optimizer, scheduler,
        early_stopping_patience=patience,
        save_path=save_path,
    )
    print(f"  Phase 1 complete — best val: {best_val:.4f}  →  {save_path}")
    return save_path


# ── Phase 2 — full model ──────────────────────────────────────────────────────

def run_phase2(model, train_loader, val_loader, fold, patience) -> float:
    """
    Unfreeze all layers with discriminative LRs per ConvNeXt stage.
    Early stages barely touched, late stages moderate, head continues at 1e-4.
    Saves checkpoint to ./checkpoints/fold{fold}_phase2.pth
    """
    save_path = f"./checkpoints/fold{fold}_phase2.pth"
    print(f"\n  [Fold {fold}] Phase 2 — full model, discriminative LRs")

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    def stage_params(stages):
        return [p for n, p in model.named_parameters()
                if any(f'stages.{s}.' in n for s in stages)]

    optimizer = optim.AdamW([
        {'params': stage_params([0, 1]), 'lr': 5e-6},  # edges/textures — barely touch
        {'params': stage_params([2, 3]), 'lr': 2e-5},  # semantics — moderate
        {'params': list(model.head.parameters()), 'lr': 1e-4},  # head — already warmed up
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val = train(
        model, train_loader, val_loader,
        optimizer, scheduler,
        early_stopping_patience=patience,
        save_path=save_path,
    )
    print(f"  Phase 2 complete — best val: {best_val:.4f}  →  {save_path}")
    return best_val


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_loader(dataset: ImageDataset, labels_int: np.ndarray,
                cloud: bool, train: bool = True) -> DataLoader:
    if train:
        class_freqs = np.bincount(labels_int) / len(labels_int)
        class_weights = 1.0 / class_freqs
        sample_weights = class_weights[labels_int]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
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
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=train,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(data_dir: str, img_size: int, cloud: bool,
         phase: int, checkpoint: str, n_folds: int, patience: int) -> None:
    # ── Load CSVs ──────────────────────────────────────────────────────────────
    df_features = pd.read_csv(f"{data_dir}/train_features.csv").set_index('id')
    df_labels = pd.read_csv(f"{data_dir}/train_labels.csv").set_index("id")
    df = pd.concat([df_features, df_labels], axis=1)

    all_paths = (data_dir + "/" + df['filepath']).tolist()
    label_cols = df_labels.columns.tolist()
    labels_onehot = df[label_cols].values.astype(np.float32)  # (N, 8)
    labels_int = np.argmax(labels_onehot, axis=1)  # (N,)

    # ── Detect site column ─────────────────────────────────────────────────────
    site_col = None
    for candidate in ['site', 'location', 'camera_site', 'site_id']:
        if candidate in df.columns:
            site_col = candidate
            break
    if site_col is None:
        raise ValueError(
            f"No site column found. Available columns: {df.columns.tolist()}\n"
            "Add your site column name to the candidate list above."
        )
    groups = df[site_col].values
    print(f"Using '{site_col}' as group — {len(np.unique(groups))} unique sites")

    # ── StratifiedGroupKFold ───────────────────────────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(np.arange(len(all_paths)), labels_int, groups), start=1
    ):
        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold}/{n_folds}  —  train={len(train_idx)}  val={len(val_idx)}")

        # Sanity check — must always be empty
        overlap = set(groups[train_idx]) & set(groups[val_idx])
        print(f"  Site overlap: {'✔ none' if not overlap else f'⚠ {overlap}'}")

        train_dataset = ImageDataset(
            [all_paths[i] for i in train_idx],
            labels_onehot[train_idx], train=True, img_size=img_size
        )
        val_dataset = ImageDataset(
            [all_paths[i] for i in val_idx],
            labels_onehot[val_idx], train=False, img_size=img_size
        )

        train_loader = make_loader(train_dataset, labels_int[train_idx], cloud, train=True)
        val_loader = make_loader(val_dataset, None, cloud, train=False)

        if phase == 1:
            model = build_model(cloud)
            run_phase1(model, train_loader, val_loader, fold, patience)

        elif phase == 2:
            # Auto-resolve per-fold phase 1 checkpoint if no override given
            ckpt = checkpoint if checkpoint else f"./checkpoints/fold{fold}_phase1.pth"
            if not os.path.exists(ckpt):
                raise FileNotFoundError(
                    f"Checkpoint not found: {ckpt}\n"
                    "Run phase 1 first:  python train.py --phase 1"
                )
            print(f"  Loading: {ckpt}")
            model = load_model(ckpt, cloud)  # build_model + load_state_dict in model_utils
            best_val = run_phase2(model, train_loader, val_loader, fold, patience)
            fold_results.append(best_val)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if phase == 1:
        print("Phase 1 complete for all folds.")
        print("\nNext — run phase 2:")
        print(f"  python train.py --phase 2 --epochs {EPOCHS} --n_folds {n_folds} [other args]")
    else:
        print("Phase 2 complete — K-Fold results:")
        for i, loss in enumerate(fold_results, start=1):
            print(f"  Fold {i}: {loss:.4f}")
        print(f"  Mean: {np.mean(fold_results):.4f}  ±  {np.std(fold_results):.4f}")
        print(f"\nFinal models: ./checkpoints/fold1_phase2.pth ... fold{n_folds}_phase2.pth")
        print("Pass these to submission.py for ensemble.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--local', action='store_true', default=False)

    # Phase control
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='1 = head only  |  2 = full model (loads phase 1 checkpoints)')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Override checkpoint path for phase 2. '
                             'If omitted, auto-resolves fold{n}_phase1.pth per fold.')
    args = parser.parse_args()

    DEVICE = "cuda:0"
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    cloud = not args.local

    os.makedirs('./checkpoints', exist_ok=True)

    main(
        args.data_dir, args.img_size, cloud,
        args.phase, args.checkpoint,
        args.n_folds, args.patience,
    )
