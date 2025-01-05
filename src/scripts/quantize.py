import torch
from src.models.lightning_module import SegmentationModel
from src.models.smp_models import get_deeplab_model, get_psp_model, get_unet_model
from transformers import AutoModelForSemanticSegmentation

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


class SMPWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SMPWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class HFWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HFWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x).logits
        out = torch.nn.functional.interpolate(out, size=(512, 512), mode="nearest")
        return out


def quantize(model, name):
    example_input = torch.rand(1, 3, 512, 512)
    model = model.cpu()
    # scripted_model = torch.jit.trace(model, example_input)
    # scripted_model.save(f"data/quantized/{name}.pt")

    # optimized = optimize_for_mobile(scripted_model)
    # optimized._save_for_lite_interpreter(f"data/quantized/{name}_optimized.ptl")
    torch.onnx.export(
        model, example_input, f"data/quantized/{name}.onnx", opset_version=11
    )


def quantize_smp() -> None:
    for model_fn, arch, backbone, checkpoint in smp_models:
        model = model_fn(backbone, num_classes=21)
        model = SegmentationModel.load_from_checkpoint(checkpoint, model=model.model)
        model = SMPWrapper(model.model)
        model.eval()

        print(f"Quantizing {arch} with {backbone} backbone")
        quantize(model, f"{arch}_{backbone}")


def quantize_hf() -> None:
    for arch, model_name, checkpoint in hf_models:
        model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
        model = HFWrapper(model)
        model.eval()

        print(f"Quantizing {arch} with {model_name} backbone")
        quantize(model, f"{arch}_{model_name}")


def main() -> None:
    quantize_smp()
    quantize_hf()


if __name__ == "__main__":
    main()
