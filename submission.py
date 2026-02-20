import numpy as np
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
from model_utils import load_model
from aug_utils import predict_with_tta
from bbox_utils import load_megadetector_results, crop_to_bbox

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = 512
N_FOLDS = 5
DATA_DIR = Path("./data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLS = [
    "antelope_duiker", "bird", "blank",
    "civet_genet", "hog", "leopard",
    "monkey_prosimian", "rodent"
]

# ── Load all fold models (phase 2 checkpoints) ────────────────────────────────
print("Loading fold models...")
models = []
for fold in range(1, N_FOLDS + 1):
    ckpt = f"./checkpoints/fold{fold}_phase2.pth"
    model = load_model(ckpt, cloud=False)  # load_model handles arch + weights + prefix strip
    model.eval()
    models.append(model)
print(f"Loaded {len(models)} fold models")

# ── Load MegaDetector bboxes for test set ────────────────────────────────────
bbox_map = load_megadetector_results("./data/megadetector_test.json", base_dir="./data/test_features")
print(f"Loaded bboxes for {len(bbox_map)} test images")

# ── Load test file paths ───────────────────────────────────────────────────────
df_test = pd.read_csv(DATA_DIR / "test_features.csv").set_index("id")
img_ids = df_test.index.tolist()
img_paths = [DATA_DIR / fp for fp in df_test["filepath"].tolist()]

# ── Inference — bbox crop → TTA → ensemble across folds ───────────────────────
rows = []

for img_id, img_path in tqdm(zip(img_ids, img_paths), total=len(img_ids), desc="Predicting"):
    img_np = np.array(Image.open(img_path).convert("RGB"))  # HWC uint8

    # Crop to animal bbox — same preprocessing as training dataset
    bbox = bbox_map.get(Path(img_path).name)
    img_np = crop_to_bbox(img_np, bbox, margin=0.15)

    # TTA over each fold model, then average
    fold_probs = []
    for model in models:
        # predict_with_tta handles all transforms internally
        # pass the cropped numpy image — NOT a tensor
        probs = predict_with_tta(model, img_np, DEVICE, image_size=IMG_SIZE)
        fold_probs.append(probs.cpu())

    # Average across folds — all are valid probability vectors, mean stays valid
    ensemble_probs = torch.stack(fold_probs).mean(dim=0)
    rows.append([img_id, *ensemble_probs.numpy()])

# ── Build submission DataFrame ────────────────────────────────────────────────
df_submission = pd.DataFrame(rows, columns=["id"] + LABEL_COLS).set_index("id")

# Sanity check — all rows must sum to ~1.0
row_sums = df_submission.sum(axis=1)
assert (row_sums - 1.0).abs().max() < 1e-4, \
    f"Probabilities don't sum to 1! Max deviation: {(row_sums - 1.0).abs().max():.6f}"

df_submission.to_csv("./submission.csv", index=True)

print(df_submission.head())
print(f"\nShape            : {df_submission.shape}")
print(f"Probability sums : min={row_sums.min():.6f}  max={row_sums.max():.6f}")
