import os
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ADE20KSegmentation(Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["training", "validation"] = "training",
        transform=None,
        target_transform=None,
        percent=1.0,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.image_dir = os.path.join(
            self.root, "ADEChallengeData2016/images", self.split
        )
        self.mask_dir = os.path.join(
            self.root, "ADEChallengeData2016/annotations", self.split
        )

        num_files = int(len(os.listdir(self.image_dir)) * percent)

        self.image_files = sorted(os.listdir(self.image_dir))[:num_files]
        self.mask_files = sorted(os.listdir(self.mask_dir))[:num_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert(
            "RGB"
        )
        mask = Image.open(os.path.join(self.mask_dir, self.mask_files[idx]))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask
