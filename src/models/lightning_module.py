import pytorch_lightning as pl
import torch
from torchmetrics.classification import MulticlassJaccardIndex


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4, num_classes=21):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.mean_iou = MulticlassJaccardIndex(num_classes, ignore_index=255)

        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)

        masks = masks.long()

        loss = self.criterion(outputs, masks)
        self.log("train_loss", loss)
        return loss

    def run_metrics(self, outputs, masks):
        masks = masks.long()

        loss = self.criterion(outputs, masks)
        self.log("val_loss", loss)
        miou = self.mean_iou(torch.argmax(outputs, dim=1), masks)
        self.log("val_miou", miou)
        elapsed_time = self.start_time.elapsed_time(self.end_time)
        self.log("val_time", elapsed_time)

    def validation_step(self, batch, batch_idx):
        images, masks = batch

        self.start_time.record()
        outputs = self.model(images)
        self.end_time.record()

        torch.cuda.synchronize()
        self.run_metrics(outputs, masks)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


class SegmentationTransformerModel(SegmentationModel):
    def __init__(self, model, learning_rate=1e-4, num_classes=21):
        super(SegmentationTransformerModel, self).__init__(
            model, learning_rate, num_classes
        )

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        images, masks = batch

        self.start_time.record()
        outputs = self.model(images).logits
        outputs = torch.nn.functional.interpolate(
            outputs, size=(masks.shape[1], masks.shape[2]), mode="nearest"
        )
        self.end_time.record()

        torch.cuda.synchronize()
        self.run_metrics(outputs, masks)
