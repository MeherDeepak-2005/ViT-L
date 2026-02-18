from torch.utils.data import Dataset
from PIL import Image
import torch as th
import pandas as pd
from torchvision import transforms
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, data_dir: str):
        df_features = pd.read_csv(f"{data_dir}/train_features.csv").set_index('id')
        df_labels = pd.read_csv(f"{data_dir}/train_labels.csv").set_index("id")

        self.df = pd.concat([df_features, df_labels], axis=1)
        self.transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        self.labels = th.from_numpy(
            self.df[df_labels.columns].values
        )


        # saving images in ram (500 mb only)
        image_paths = self.df['filepath'].values
        self.imgs = []
        for image_path in tqdm(image_paths):
            img = Image.open(f"{data_dir}/{image_path}").convert('RGB')
            self.imgs.append(img)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        tensors = self.transforms(img)
        label = self.labels[idx]

        return tensors, label