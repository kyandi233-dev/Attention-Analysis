"""
This script runs YOLOv8 to generate a report (CSV) of all detections found in a folder with images

Author: Ignacio Hernández Montilla, 2023
"""

import os
from pathlib import Path
import argparse

import pandas as pd
from ultralytics import YOLO
import supervision as spv
from utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--path_model', type=str, help="Path to the model")
    parser.add_argument("-p", '--path_data', type=str, help="Path to the data")
    parser.add_argument("-o", '--path_output', type=str, default="runs/reports", help="The output will be saved here")
    parser.add_argument("--show", action="store_true", help="Show the predictions")
    parser.add_argument("--frame_time", type=int, default=30, help="Duration (ms) of each frame")
    args = parser.parse_args()

    # Loading the model
    try:
        path_model = Path(args.path_model)
        model = YOLO(path_model)
    except FileNotFoundError:
        print("ERROR: Could not load the YOLO model")
        exit()

    # Get the results
    if args.path_data:
        path_output = Path(args.path_output)
        path_output.mkdir(exist_ok=True, parents=True)
        path_report = path_output / "report.csv"
        report = pd.DataFrame(columns=['image_name', 'detection', 'x1', 'y1', 'x2', 'y2'])

        class_colors = spv.ColorPalette.from_hex(['#ffff66', '#66ffcc', '#ff99ff', '#ffcc99'])
        class_names_dict = model.model.names
        bbox_annotator = spv.BoundingBoxAnnotator(thickness=2, color=class_colors)
        label_annotator = spv.LabelAnnotator(color=class_colors, text_color=spv.Color.from_hex("#000000"))

        for f in os.listdir(args.path_data):
            img = cv2.imread(os.path.join(args.path_data, f))
            img, _ = smart_resize(img, new_size=640)
            result = model(img, agnostic_nms=True, verbose=False)[0]
            detections = spv.Detections.from_ultralytics(result)

            for i, bbox in enumerate(detections.xyxy):
                x1, y1, x2, y2 = bbox.astype(int)
                label = class_names_dict[detections.class_id[i]]
                report.loc[len(report), :] = [f, label, x1, y1, x2, y2]

            if args.show:
                img = annotate_frame(img, detections, bbox_annotator, label_annotator, class_names_dict)
                cv2.imshow("Face parts", img)
                k = cv2.waitKey(args.frame_time)

        if args.show:
            cv2.destroyAllWindows()
        report.to_csv(path_report, index=False)
        print("Report saved to ", str(path_report))
    else:
        print("ERROR: No data folder (path_data) provided")
        exit()
