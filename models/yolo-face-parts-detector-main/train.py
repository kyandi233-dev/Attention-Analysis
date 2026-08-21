"""
Train your YOLOv8 models with this script
(You should have downloaded all the data and run prepare_full_dataset.py first)

Author: Ignacio Hernández Montilla, 2023
"""

from ultralytics import YOLO
from pathlib import Path
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", '--arch', type=str, default='n', help="Architecture (n, s, m, l, x)")
    parser.add_argument("-n", '--name', type=str, default="train", help="Run name")
    parser.add_argument("-d", '--path_data', type=str, help="Path to the datasets folder")
    parser.add_argument("-i", '--image_size', type=int, default=640, help="Image size")
    parser.add_argument("-b", '--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument("-e", '--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument("--device", type=str, default=['0'], nargs='+', help="Device list (also accepts 'cpu')")
    args = parser.parse_args()

    if args.path_data is not None:
        path_datasets = Path(args.path_data)
    else:
        path_datasets = Path.home() / "Documents" / "Datasets"

    path_face_parts = path_datasets / "Face-Parts-Dataset"
    path_yaml = path_face_parts / "split" / "data.yaml"

    # Training
    model = YOLO("weights/yolov8{}.pt".format(args.arch))
    results = model.train(data=str(path_yaml), task="detect", name="{}_{}".format(args.name, args.arch),
                          epochs=args.epochs, imgsz=args.image_size, batch=args.batch_size,
                          device=",".join(args.device),
                          scale=0.25, degrees=25.0, mosaic=0.8)
