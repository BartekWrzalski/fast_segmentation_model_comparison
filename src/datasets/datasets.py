import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, VOCSegmentation

from src.datasets.ade20k import ADE20KSegmentation

CITYSCAPES_35_TO_19 = {
    0: 255,  # Unlabeled
    1: 255,  # Ego vehicle
    2: 255,  # Rectification border
    3: 255,  # Out of ROI
    4: 255,  # Static
    5: 255,  # Dynamic
    6: 255,  # Ground
    7: 0,  # Road
    8: 1,  # Sidewalk
    9: 255,  # Building
    10: 255,  # Wall
    11: 2,  # Fence
    12: 3,  # Pole
    13: 4,  # Traffic light
    14: 255,  # Traffic sign
    15: 255,  # Vegetation fence
    16: 255,  # Vegetation
    17: 5,  # Terrain
    18: 255,  # Sky
    19: 6,  # Person
    20: 7,  # Rider
    21: 8,  # Car
    22: 9,  # Truck
    23: 10,  # Bus
    24: 11,  # Train
    25: 12,  # Motorcycle
    26: 13,  # Bicycle
    27: 14,  # License plate
    28: 15,  # Rail track
    29: 255,  # Other void classes
    30: 255,  # Crowd
    31: 16,  # Reflection
    32: 17,  # Other moving
    33: 18,  # Fire hydrant
    -1: 255,  # Parking meter
}


def __get_dataloader(
    root,
    batch_size,
    num_workers,
    transform,
    target_transform,
    dataset_cls,
    shuffle=True,
    **kwargs,
):
    if transform is None:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.ToTensor(),
            ]
        )

    if target_transform is None:
        target_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((512, 512), interpolation=Image.NEAREST),
                torchvision.transforms.ToTensor(),
            ]
        )

    dataset = dataset_cls(
        root=root,
        transform=transform,
        target_transform=target_transform,
        **kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )


def get_ade_dataloaders(
    root: str = "data/ade",
    batch_size: int = 4,
    num_workers: int = 4,
    transform=None,
    mask_transform=None,
    **kwargs,
):
    if mask_transform is None:
        mask_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((512, 512), interpolation=Image.NEAREST),
            ]
        )

    args = {
        "root": root,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "transform": transform,
        "mask_transform": mask_transform,
        "dataset_cls": ADE20KSegmentation,
        **kwargs,
    }

    train_dl = __get_dataloader(
        **args,
        split="training",
    )
    val_dl = __get_dataloader(
        **args,
        split="validation",
        shuffle=False,
    )
    return train_dl, val_dl


def get_citys_dataloader(
    root: str = "data/cityscapes",
    batch_size: int = 4,
    num_workers: int = 4,
    transform=None,
    mask_transform=None,
    **kwargs,
):
    if mask_transform is None:

        def mask_transform(mask):
            mask = torchvision.transforms.functional.resize(
                mask,
                (512, 512),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            )
            mask = torch.from_numpy(
                np.vectorize(CITYSCAPES_35_TO_19.get)(np.array(mask))
            ).long()
            return mask

    args = {
        "root": root,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "transform": transform,
        "target_transform": mask_transform,
        "dataset_cls": Cityscapes,
        "target_type": "semantic",
        "mode": "fine",
        **kwargs,
    }

    train_dl = __get_dataloader(
        **args,
        split="train",
    )
    val_dl = __get_dataloader(
        **args,
        split="val",
        shuffle=False,
    )
    return train_dl, val_dl


def get_voc_dataloader(
    root: str = "data/voc",
    batch_size: int = 4,
    num_workers: int = 4,
    transform=None,
    mask_transform=None,
    **kwargs,
):
    if mask_transform is None:

        def mask_transform(mask):
            mask = torchvision.transforms.functional.resize(
                mask,
                (512, 512),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            )
            mask = torch.from_numpy(np.array(mask)).long()
            return mask

    args = {
        "root": root,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "transform": transform,
        "target_transform": mask_transform,
        "dataset_cls": VOCSegmentation,
        **kwargs,
    }

    train_dl = __get_dataloader(
        **args,
        image_set="train",
    )
    val_dl = __get_dataloader(
        **args,
        image_set="val",
        shuffle=False,
    )
    return train_dl, val_dl
