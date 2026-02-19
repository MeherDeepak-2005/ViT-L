import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from aug_utils import get_train_transforms, get_val_transforms


class ImageDataset(Dataset):
    def __init__(self, img_paths: list, labels: np.ndarray, train: bool = True, img_size: int = 512):
        """
        img_paths : list of file path strings
        labels    : (N, 8) one-hot numpy array  OR  (N,) integer numpy array
        train     : use train transforms if True, val transforms if False
        """
        self.transforms = get_train_transforms(img_size) if train else get_val_transforms(img_size)

        # Handle both one-hot (N, 8) and integer (N,) labels
        if labels.ndim == 2:
            # one-hot → integer class indices
            self.labels = torch.from_numpy(np.argmax(labels, axis=1)).long()
        else:
            self.labels = torch.from_numpy(labels).long()

        # Load all images into RAM as numpy arrays (albumentations needs numpy HWC uint8)
        self.imgs = []
        for image_path in tqdm(img_paths, desc="Loading images"):
            img = np.array(Image.open(image_path).convert('RGB'))  # HWC uint8
            self.imgs.append(img)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_np = self.imgs[idx]  # HWC uint8 numpy
        tensor = self.transforms(image=img_np)['image']  # CHW float32 torch
        label = self.labels[idx]  # scalar LongTensor
        return tensor, label
