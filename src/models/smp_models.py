import segmentation_models_pytorch as smp

from src.models.lightning_module import SegmentationModel


def get_unet_model(model_name: str, num_classes: int):
    model = smp.Unet(
        encoder_name=model_name,
        encoder_weights="imagenet",
        classes=num_classes,
    )
    model = SegmentationModel(model, num_classes=num_classes)
    return model


def get_deeplab_model(model_name: str, num_classes: int):
    model = smp.DeepLabV3Plus(
        encoder_name=model_name,
        encoder_weights="imagenet",
        classes=num_classes,
    )
    model = SegmentationModel(model, num_classes=num_classes)
    return model


def get_psp_model(model_name: str, num_classes: int):
    model = smp.PSPNet(
        encoder_name=model_name,
        encoder_weights="imagenet",
        classes=num_classes,
    )
    model = SegmentationModel(model, num_classes=num_classes)
    return model
