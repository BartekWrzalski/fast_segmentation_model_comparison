from argparse import ArgumentParser

import torch

from src.datasets.datasets import get_voc_dataloader
from src.models.smp_models import get_deeplab_model, get_psp_model, get_unet_model
from src.trainer import get_default_trainer

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["unet", "psp", "deeplab"],
    )
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def train_model(model, args):
    train_dl, val_dl = get_voc_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = get_default_trainer(
        max_epochs=args.max_epochs,
        model_name=args.model,
        quiet=args.quiet,
        accelerator=args.accelerator,
        data_dir=args.data_dir,
    )

    trainer.fit(model, train_dl, val_dl)


def main():
    args = parse_args()

    fn = {
        "unet": get_unet_model,
        "psp": get_psp_model,
        "deeplab": get_deeplab_model,
    }[args.model]

    model = fn(args.backbone, num_classes=21)

    train_model(model, args)


if __name__ == "__main__":
    main()
