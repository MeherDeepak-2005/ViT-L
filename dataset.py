from torch.utils.data import Dataset
from PIL import Image
import torch as th
import pandas as pd
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self):
        df_features = pd.read_csv("./data/train_features.csv").set_index('id')
        df_labels = pd.read_csv("./data/train_labels.csv").set_index("id")

        self.df = pd.concat([df_features, df_labels], axis=1)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        self.labels = th.from_numpy(
            self.df[df_labels.columns].values
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df['filepath'].iloc[idx]
        img = Image.open(f"./data/{image_path}").convert('RGB')
        tensors = self.transforms(img)
        label = self.labels[idx]

        return tensors, label