import os

import requests
import yaml


def craete_data_directories(directory: str):
    os.makedirs(directory, exist_ok=True)


def download_coco_dataset(data_type: str, data_zip: str):
    coco_dataset_url = f"http://images.cocodataset.org/zips/{data_type}2017.zip"

    if not os.path.exists(data_zip):
        print(f"Downloading COCO {data_type} dataset...")
        os.system(f"wget {coco_dataset_url} -O {data_zip}")
        print(f"COCO {data_type} dataset downloaded.")
    else:
        print(f"COCO {data_type} dataset is already downloaded.")


def download_coco_annotations(data_zip: str):
    coco_annotations_url = (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )

    if not os.path.exists(data_zip):
        print("Downloading COCO annotations...")
        os.system(f"wget {coco_annotations_url} -O {data_zip}")
        print("COCO annotations downloaded.")
    else:
        print("COCO annotations are already downloaded.")


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["download_data"]
    craete_data_directories(params["out_dir"])

    download_coco_dataset("train", params["train_data_zip"])
    download_coco_dataset("val", params["val_data_zip"])
    download_coco_annotations(params["annotations_zip"])
