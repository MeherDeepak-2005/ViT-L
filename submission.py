import numpy as np
import timm
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from aug_utils import predict_with_tta

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = 512  # must match training resolution
BATCH_SIZE = 32
MODEL_PATH = "./models/convnext_large.pth"
DATA_DIR = Path("./data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLS = [
    "antelope_duiker", "bird", "blank",
    "civet_genet", "hog", "leopard",
    "monkey_prosimian", "rodent"
]

# ── Build & load model ────────────────────────────────────────────────────────
model = timm.create_model("convnext_large_in22k", pretrained=False)
model.head.fc = nn.Linear(model.head.fc.in_features, out_features=8)

# Strip torch.compile prefix if model was compiled during training
weights = torch.load(MODEL_PATH, map_location=DEVICE)
cleaned = {
    k.replace('_orig_mod.', ''): v
    for k, v in weights.items()
}
model.load_state_dict(cleaned)
model = model.to(DEVICE, memory_format=torch.channels_last)
model.eval()

# ── Load test file paths from CSV (consistent ordering, no stray files) ───────
df_test = pd.read_csv(DATA_DIR / "test_features.csv").set_index("id")
img_ids = df_test.index.tolist()
img_paths = [DATA_DIR / fp for fp in df_test["filepath"].tolist()]

# ── Inference with TTA ────────────────────────────────────────────────────────
rows = []

for img_id, img_path in tqdm(zip(img_ids, img_paths), total=len(img_ids), desc="Predicting"):
    # Load as raw numpy HWC uint8 — predict_with_tta handles all transforms internally
    img_np = np.array(Image.open(img_path).convert("RGB"))

    # Returns (8,) tensor of probabilities summing to 1
    probs = predict_with_tta(model, img_np, DEVICE, image_size=IMG_SIZE)

    rows.append([img_id, *probs.cpu().numpy()])

# ── Build submission DataFrame ────────────────────────────────────────────────
df_submission = pd.DataFrame(rows, columns=["id"] + LABEL_COLS)
df_submission = df_submission.set_index("id")

# Sanity check — all rows should sum to ~1.0
row_sums = df_submission.sum(axis=1)
assert (row_sums - 1.0).abs().max() < 1e-4, "Probabilities don't sum to 1!"

df_submission.to_csv("./submission.csv", index=True)
print(df_submission.head())
print(f"\nSubmission shape : {df_submission.shape}")
print(f"Probability sums : min={row_sums.min():.6f}  max={row_sums.max():.6f}")
