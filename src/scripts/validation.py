import json

import torch
from transformers import AutoModelForSemanticSegmentation

from src.datasets.datasets import get_voc_dataloader
from src.models.lightning_module import SegmentationModel, SegmentationTransformerModel
from src.models.smp_models import get_deeplab_model, get_psp_model, get_unet_model
from src.validate import validate

torch.set_float32_matmul_precision("high")

smp_models = [
    (
        get_unet_model,
        "unet",
        "resnet18",
        "/home/bartek/OKNO_segmentation_model_comparison/data/models/unet_resnet18/epoch=18-step=1748.ckpt",
    ),
    (
        get_unet_model,
        "unet",
        "timm-mobilenetv3_large_100",
        "/home/bartek/OKNO_segmentation_model_comparison/data/models/unet_timm-mobilenetv3_large_100/epoch=41-step=3864.ckpt",
    ),
    (
        get_psp_model,
        "psp",
        "efficientnet-b0",
        "/home/bartek/OKNO_segmentation_model_comparison/data/models/pspnet_efficientnet-b0/epoch=78-step=7268.ckpt",
    ),
    (
        get_psp_model,
        "psp",
        "timm-mobilenetv3_large_100",
        "/home/bartek/OKNO_segmentation_model_comparison/data/models/pspnet_timm-mobilenetv3_large_100/epoch=68-step=6348.ckpt",
    ),
    (
        get_deeplab_model,
        "deep",
        "timm-mobilenetv3_large_100",
        "/home/bartek/OKNO_segmentation_model_comparison/data/models/deeplabv3_timm-mobilenetv3_large_100/epoch=15-step=1472.ckpt",
    ),
]

hf_models = [
    ("deep", "vit", "apple/deeplabv3-mobilevit-small"),
    ("deep", "vit-x", "apple/deeplabv3-mobilevit-x-small"),
    ("deep", "vit-xx", "apple/deeplabv3-mobilevit-xx-small"),
]


def validate_smp() -> dict:
    all_metrics = {}
    for model_fn, arch, backbone, checkpoint in smp_models:
        model = model_fn(backbone, num_classes=21)
        model = SegmentationModel.load_from_checkpoint(checkpoint, model=model.model)
        model.eval()

        _, val_dl = get_voc_dataloader(
            batch_size=1,
            num_workers=4,
        )
        print(f"Validating {arch} with {backbone} backbone")
        metrics = validate(model, val_dl, f"{arch}_{backbone}")
        metrics["val_loss"] = metrics["val_loss"].item()
        metrics["val_miou"] = metrics["val_miou"].item()
        metrics["val_time"] = metrics["val_time"].item()
        metrics["size"] = sum(p.numel() for p in model.parameters())
        all_metrics[f"{arch}_{backbone}"] = metrics

    return all_metrics


def validate_hf() -> dict:
    all_metrics = {}
    for arch, backbone, checkpoint in hf_models:
        model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
        model = SegmentationTransformerModel(model=model, num_classes=21)
        model.eval()

        _, val_dl = get_voc_dataloader(
            batch_size=1,
            num_workers=4,
        )
        print(f"Validating {arch} with {backbone} backbone")
        metrics = validate(model, val_dl, f"{arch}_{backbone}")
        metrics["val_loss"] = metrics["val_loss"].item()
        metrics["val_miou"] = metrics["val_miou"].item()
        metrics["val_time"] = metrics["val_time"].item()
        metrics["size"] = sum(p.numel() for p in model.parameters())
        all_metrics[f"{arch}_{backbone}"] = metrics

    return all_metrics


def main():
    smp_metrics = validate_smp()
    hf_metrics = validate_hf()

    all_metrics = {**smp_metrics, **hf_metrics}
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)


if __name__ == "__main__":
    main()
