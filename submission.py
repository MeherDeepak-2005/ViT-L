from torchvision.models import vit_l_16, ViT_L_16_Weights
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image

model = vit_l_16()
model.heads.head = torch.nn.Linear(1024, 8)
weights = torch.load("./models/ViT-L_epoch-9.pth")

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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
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

        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1).cpu().tolist()

        for img_id, prob_label in zip(img_ids, probs):
            rows.append([img_id, *prob_label])

df = pd.DataFrame(rows)
columns = "id","antelope_duiker","bird","blank","civet_genet","hog","leopard","monkey_prosimian","rodent"
df.columns = columns
df.to_csv("./submissions.csv")
print(df.head())

