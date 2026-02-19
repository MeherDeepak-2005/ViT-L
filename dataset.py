from torch.utils.data import Dataset
from PIL import Image
import torch as th
from torchvision import transforms
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, img_paths: list, labels, transforms: transforms.Compose):
        self.transforms = transforms

        self.labels = th.from_numpy(
            labels
        )

        # saving images in ram (500 mb only)
        self.imgs = []
        for image_path in tqdm(img_paths):
            img = Image.open(image_path).convert('RGB')
            self.imgs.append(img)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        tensors = self.transforms(img)
        label = self.labels[idx]

        return tensors, label
