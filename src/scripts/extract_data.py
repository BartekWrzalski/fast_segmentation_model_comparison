import os

import yaml


def craete_data_directories(directory: str):
    os.makedirs(directory, exist_ok=True)


def extract_coco_dataset(source: str, destination: str):
    print("Extracting COCO dataset...")
    os.system(f"unzip {source} -d {destination}")
    print("COCO dataset extracted.")


def extract_coco_annotations(source: str, destination: str):
    print("Extracting COCO annotations...")
    os.system(f"unzip {source} -d {destination}")
    print("COCO annotations extracted.")


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["extract_data"]
    craete_data_directories(params["out_dir"])

    extract_coco_dataset(params["train_data"]["source"], params["out_dir"])
    extract_coco_dataset(params["val_data"]["source"], params["out_dir"])
    extract_coco_annotations(params["annotations"]["source"], params["out_dir"])
