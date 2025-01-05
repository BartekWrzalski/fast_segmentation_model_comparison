import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


def get_default_trainer(
    max_epochs: int,
    model_name: str,
    quiet: bool = False,
    accelerator: str = "auto",
    data_dir: str = "data",
) -> pl.Trainer:
    logger = TensorBoardLogger(
        save_dir=f"{data_dir}/logs",
        name=model_name,
        default_hp_metric=False,
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=f"{data_dir}/models/{model_name}",
            verbose=False,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=8,
            verbose=False,
        ),
    ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        accelerator=accelerator,
        enable_progress_bar=not quiet,
    )

    return trainer
