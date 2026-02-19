import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from aug_utils import get_train_transforms, get_val_transforms
from bbox_utils import load_megadetector_results, crop_to_bbox


class ImageDataset(Dataset):
    def __init__(self, img_paths, labels, train=True, img_size=512,
                 bbox_json=None):  # ← new
        self.transforms = get_train_transforms(img_size) if train else get_val_transforms(img_size)
        self.bbox_map = load_megadetector_results(bbox_json, base_dir="./data/train_features") if bbox_json else {}

        if labels.ndim == 2:
            self.labels = torch.from_numpy(np.argmax(labels, axis=1)).long()
        else:
            self.labels = torch.from_numpy(labels).long()

        self.imgs = []
        for image_path in tqdm(img_paths, desc="Loading images"):
            img_np = np.array(Image.open(image_path).convert('RGB'))

            # crop to animal if bbox available
            bbox = self.bbox_map.get(str(image_path))
            img_np = crop_to_bbox(img_np, bbox, margin=0.15)

            self.imgs.append(img_np)

    def __getitem__(self, idx):
        tensor = self.transforms(image=self.imgs[idx])['image']
        return tensor, self.labels[idx]
