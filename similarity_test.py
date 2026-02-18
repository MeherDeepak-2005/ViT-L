import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "./data"
TRAIN_DIR = f"{DATA_DIR}/train_features"
TEST_DIR  = f"{DATA_DIR}/test_features"
MODEL_PATH = "./models/ViT-L_epoch-9.pth"

BATCH_SIZE = 32
NUM_WORKERS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# TRANSFORMS (ViT-L standard)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

import os
from PIL import Image
from torch.utils.data import Dataset

class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.images = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0  # dummy label

# -----------------------------
# DATASETS
# -----------------------------
train_dataset = FlatImageDataset(TRAIN_DIR, transform=transform)
test_dataset  = FlatImageDataset(TEST_DIR, transform=transform)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# -----------------------------
# MODEL (vit_l_16)
# -----------------------------
model = models.vit_l_16(weights=models.ViT_L_16_Weights)

model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# -----------------------------
# EMBEDDING EXTRACTION
# -----------------------------
def extract_embeddings(dataloader):
    all_embeddings = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(DEVICE)

            # ---- Forward manually through ViT ----
            x = model._process_input(images)          # patch + pos embed
            n = x.shape[0]

            batch_class_token = model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = model.encoder(x)

            cls_embedding = x[:, 0]  # CLS token

            all_embeddings.append(cls_embedding.cpu())

    return torch.cat(all_embeddings, dim=0)


print("\n🔹 Extracting train embeddings...")
train_emb = extract_embeddings(train_loader)

print("\n🔹 Extracting test embeddings...")
test_emb = extract_embeddings(test_loader)

print(f"\nTrain embeddings shape: {train_emb.shape}")
print(f"Test embeddings shape:  {test_emb.shape}")

# -----------------------------
# NORMALIZE
# -----------------------------
train_emb = F.normalize(train_emb, dim=1)
test_emb  = F.normalize(test_emb, dim=1)

print("✅ Embeddings normalized")

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
similarity_matrix = test_emb @ train_emb.T

print(f"Similarity matrix shape: {similarity_matrix.shape}")

# -----------------------------
# MAX SIMILARITY PER TEST IMAGE
# -----------------------------
max_sim, nn_idx = similarity_matrix.max(dim=1)

print("\n📊 Similarity Stats:")
print("Mean max similarity:", max_sim.mean().item())
print("Min similarity:", max_sim.min().item())
print("Max similarity:", max_sim.max().item())

# -----------------------------
# OPTIONAL: SAVE RESULTS
# -----------------------------
torch.save({
    "similarity_matrix": similarity_matrix,
    "max_similarity": max_sim,
    "nearest_neighbor_idx": nn_idx
}, "cosine_similarity_results.pth")

print("\n✅ Results saved → cosine_similarity_results.pth")
