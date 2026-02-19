import timm
import torch.nn as nn
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
from aug_utils import predict_with_tta

model = timm.create_model("convnext_large_in22k", pretrained=True)
model.head.fc = nn.Linear(in_features=model.head.fc.in_features, out_features=8)

# noinspection PyArgumentList
model = model.to(device='cuda', memory_format=torch.channels_last)

weights = torch.load("./models/convnext_large.pth")

# Remove '_orig_mod.' prefix from all keys
cleaned_state_dict = {}
for key, value in weights.items():
    if key.startswith('_orig_mod.'):
        cleaned_key = key.replace('_orig_mod.', '')
        cleaned_state_dict[cleaned_key] = value
    else:
        cleaned_state_dict[key] = value

# Load into model
model.load_state_dict(cleaned_state_dict)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval().to(device)

batch_size = 32
rows = []

test_dir = Path("./data/test_features")

imgs = list(test_dir.iterdir())

with torch.no_grad():
    for i in range(0, len(imgs), batch_size):
        batch_files = imgs[i:i + batch_size]

        img_ids = []
        tensors = []

        for img_path in batch_files:
            img_ids.append(img_path.stem)

            img = Image.open(img_path).convert("RGB")
            tensors.append(transform(img))

        batch_tensor = torch.stack(tensors).to(device)

        probs = predict_with_tta(model, batch_tensor, device)

        for img_id, prob_label in zip(img_ids, probs):
            rows.append([img_id, *prob_label])

df = pd.DataFrame(rows)
columns = "id", "antelope_duiker", "bird", "blank", "civet_genet", "hog", "leopard", "monkey_prosimian", "rodent"
df.columns = columns
df.to_csv("./submissions.csv", index=True)
print(df.head())
