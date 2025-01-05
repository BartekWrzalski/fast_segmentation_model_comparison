import ssl
from collections import namedtuple

import torch
import yaml

from src.models.smp_models import get_deeplab_model, get_psp_model, get_unet_model
from src.scripts.train import train_model

ssl._create_default_https_context = ssl._create_unverified_context
torch.set_float32_matmul_precision("high")


def get_args(model, backbone):
    Args = namedtuple(
        "args",
        [
            "model",
            "backbone",
            "max_epochs",
            "batch_size",
            "num_workers",
            "data_dir",
            "accelerator",
            "quiet",
        ],
    )

    args = Args(
        model=f"{model}_{backbone}",
        backbone=backbone,
        max_epochs=100,
        batch_size=16,
        num_workers=4,
        data_dir="data",
        accelerator="auto",
        quiet=False,
    )
    return args


def main():
    conf = yaml.safe_load(open("config.yaml"))

    for model, backbones in conf.items():
        for backbone in backbones:
            args = get_args(model, backbone)
            fn = {
                "unet": get_unet_model,
                "pspnet": get_psp_model,
                "deeplabv3": get_deeplab_model,
            }[model]

            _model = fn(backbone, num_classes=21)
            print(args.model)
            train_model(_model, args)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # Włącz benchmarkowanie
    main()
