import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.distributions import Beta


# ─────────────────────────────────────────────
# 1. PER-IMAGE TRANSFORMS (inside Dataset)
#    Input:  numpy HWC uint8 image (from cv2/PIL)
#    Output: torch tensor CHW float32, normalized
# ─────────────────────────────────────────────

def get_train_transforms(image_size=512):
    return A.Compose([
        # Geometry
        A.RandomResizedCrop(size=(512, 512), scale=(0.65, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),

        # Lighting — important for IR/low-light camera traps
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),

        # Sensor / environment noise
        A.OneOf([
            A.MotionBlur(blur_limit=9),
            A.GaussianBlur(blur_limit=(3, 7)),
        ], p=0.4),
        A.GaussNoise(std_range=(0.04, 0.2), p=0.5),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.ImageCompression(quality_range=(50, 100), p=0.3),

        # Occlusion — small patches, conservative for small subjects
        A.CoarseDropout(
            num_holes_range=(1, 6),  # was (1, 4)
            hole_height_range=(16, 48),  # was (8, 20) — scale with animal size
            hole_width_range=(16, 48),
            fill=128,
            p=0.4  # was 0.3
        ),

        # Normalize + convert to tensor
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()  # HWC numpy → CHW torch float32
    ])


def get_val_transforms(image_size=512):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# ─────────────────────────────────────────────
# 2. LABEL SMOOTHING
#    Call this before MixUp/CutMix to get soft labels
#    Input:  labels — torch.LongTensor (N,)
#    Output: soft_labels — torch.FloatTensor (N, n_classes)
# ─────────────────────────────────────────────

def smooth_labels(labels, n_classes=8, epsilon=0.1):
    """
    Convert integer class indices to smoothed soft labels.
    labels: (N,) LongTensor
    returns: (N, n_classes) FloatTensor
    """
    N = labels.size(0)
    soft = torch.full((N, n_classes), epsilon / n_classes, device=labels.device)
    soft.scatter_(1, labels.unsqueeze(1), 1.0 - epsilon + epsilon / n_classes)
    return soft


# ─────────────────────────────────────────────
# 3. MIXUP
#    Input:  images (N,C,H,W) float, labels (N, n_classes) float soft labels
#    Output: mixed images, mixed soft labels
# ─────────────────────────────────────────────

def mixup(images, labels, alpha=0.4):
    """
    images: (N, C, H, W) float tensor
    labels: (N, n_classes) float soft label tensor  ← use smooth_labels() first
    """
    lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample().item()
    idx = torch.randperm(images.size(0), device=images.device)

    mixed_images = lam * images + (1 - lam) * images[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_images, mixed_labels


# ─────────────────────────────────────────────
# 4. CUTMIX
#    Input:  images (N,C,H,W) float, labels (N, n_classes) float soft labels
#    Output: mixed images, mixed soft labels
# ─────────────────────────────────────────────

def cutmix(images, labels, alpha=1.0):
    """
    images: (N, C, H, W) float tensor
    labels: (N, n_classes) float soft label tensor  ← use smooth_labels() first
    """
    lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample().item()
    idx = torch.randperm(images.size(0), device=images.device)

    _, _, H, W = images.shape
    cut_ratio = (1 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()

    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

    # recompute actual lambda after boundary clamping
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed, mixed_labels


# ─────────────────────────────────────────────
# 5. MIXUP / CUTMIX ALTERNATOR
#    Randomly picks one per batch — standard in SOTA pipelines
# ─────────────────────────────────────────────

def mix_batch(images, labels, alpha=0.4, mixup_prob=0.5):
    """
    Randomly applies either MixUp or CutMix to a batch.
    images: (N, C, H, W) float tensor
    labels: (N,) LongTensor   ← converts internally, returns soft labels
    returns: mixed images (N,C,H,W), soft labels (N, n_classes)
    """
    n_classes = 8
    soft = smooth_labels(labels, n_classes=n_classes)

    if torch.rand(1).item() < mixup_prob:
        return mixup(images, soft, alpha=alpha)
    else:
        return cutmix(images, soft, alpha=alpha)


# ─────────────────────────────────────────────
# 6. GRIDMASK
#    Safer occlusion than random erasing for small subjects
#    Input/Output: torch tensor (C, H, W) or (N, C, H, W)
# ─────────────────────────────────────────────

def gridmask(images, d=120, r=0.6, p=0.5):
    """
    images: (N, C, H, W) float tensor
    d: grid unit size in pixels
    r: fraction of each grid unit to mask
    p: probability of applying per sample in batch
    """
    if torch.rand(1).item() > p:
        return images

    N, C, H, W = images.shape
    mask = torch.ones(H, W, device=images.device)
    hole = int(d * r)

    for i in range(0, H, d):
        for j in range(0, W, d):
            i_end = min(i + hole, H)
            j_end = min(j + hole, W)
            mask[i:i_end, j:j_end] = 0.0

    # apply independently to each sample in batch with probability p
    for n in range(N):
        if torch.rand(1).item() < p:
            images[n] = images[n] * mask.unsqueeze(0)

    return images


# ─────────────────────────────────────────────
# 7. MANIFOLD MIXUP (feature-space mixing)
#    Plug this into your model's forward() — see usage below
# ─────────────────────────────────────────────

def manifold_mixup_hook(features, labels, alpha=0.4):
    """
    Mix in feature space rather than pixel space.
    Call this inside your model forward() at a chosen intermediate layer.

    features: (N, D, H, W) or (N, D) intermediate feature tensor
    labels:   (N, n_classes) soft label tensor
    returns:  mixed features, mixed labels
    """
    lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample().item()
    idx = torch.randperm(features.size(0), device=features.device)

    mixed_features = lam * features + (1 - lam) * features[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_features, mixed_labels


# ─────────────────────────────────────────────
# 8. TEST TIME AUGMENTATION (TTA)
#    Call at inference — averages predictions over augmented views
# ─────────────────────────────────────────────

def get_tta_transforms(image_size=512):
    """Returns a list of deterministic TTA transforms."""
    base = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return [
        A.Compose(base),  # original
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1,
                                              contrast_limit=0.1, p=1.0)] + base),
        A.Compose([A.CLAHE(clip_limit=2.0, p=1.0)] + base),
        A.Compose([A.RandomGamma(gamma_limit=(90, 110), p=1.0)] + base),
    ]


def predict_with_tta(model, image_np, device, image_size=512):
    """
    image_np: single HWC uint8 numpy image (before any transform)
    returns:  averaged softmax probabilities (n_classes,)
    """
    tta_transforms = get_tta_transforms(image_size)
    preds = []
    model.eval()
    with torch.no_grad():
        for transform in tta_transforms:
            tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)
            logits = model(tensor)
            preds.append(torch.softmax(logits, dim=-1))
    return torch.stack(preds).mean(dim=0).squeeze(0)


# ─────────────────────────────────────────────
# 9. LOSS FUNCTION
#    Handles soft labels from MixUp/CutMix natively
# ─────────────────────────────────────────────

def soft_cross_entropy(logits, soft_labels):
    """
    logits:      (N, n_classes) raw model output
    soft_labels: (N, n_classes) float — works with hard or mixed labels
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_labels * log_probs).sum(dim=-1).mean()

# ─────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────
#
# ── Dataset ──
# class WildlifeDataset(Dataset):
#     def __init__(self, images_np, labels, train=True):
#         self.images = images_np          # list of HWC uint8 numpy arrays
#         self.labels = labels             # (N,) LongTensor
#         self.transform = get_train_transforms() if train else get_val_transforms()
#
#     def __getitem__(self, idx):
#         image = self.transform(image=self.images[idx])["image"]  # CHW float tensor
#         return image, self.labels[idx]
#
#
# ── Training Loop ──
# for images, labels in train_loader:
#     images, labels = images.to(device), labels.to(device)
#
#     # optional: gridmask before mixing
#     images = gridmask(images, d=60, r=0.6, p=0.5)
#
#     # MixUp / CutMix — returns soft labels automatically
#     images, soft_labels = mix_batch(images, labels, alpha=0.4)
#
#     optimizer.zero_grad()
#     logits = model(images)
#     loss = soft_cross_entropy(logits, soft_labels)
#     loss.backward()
#     optimizer.step()
#
#
# ── Inference with TTA ──
# probs = predict_with_tta(model, image_np, device)  # single image
