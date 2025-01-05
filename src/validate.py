import pytorch_lightning as pl

from src.trainer import get_default_trainer


def validate(
    model: pl.LightningModule, dataloader: pl.LightningDataModule, name: str
) -> dict:
    trainer = get_default_trainer(1, f"validate_{name}")
    trainer.validate(model, dataloader)
    return trainer.callback_metrics
