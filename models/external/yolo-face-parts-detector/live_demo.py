"""
This script loads a YOLO model and runs it on live camera feed

Author: Ignacio Hernández Montilla, 2023
"""

from pathlib import Path
import argparse
import time

from ultralytics import YOLO
import cv2
import numpy as np
import imageio
import supervision as spv
from utils import annotate_frame


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--path_model', type=str, help="Path to the model")
    parser.add_argument("-i", '--camera_id', type=int, help="Camera ID")
    parser.add_argument('--save_gif', type=str,
                        help="Save the video to a GIF file in given location")
    args = parser.parse_args()

    # Loading the model
    try:
        print("Loading the model")
        path_model = Path(args.path_model)
        model = YOLO(path_model)
    except FileNotFoundError:
        print("ERROR: Could not load the YOLO model")
        exit()

    # This will draw the detections
    class_colors = spv.ColorPalette.from_hex(['#ffff66', '#66ffcc', '#ff99ff', '#ffcc99'])
    class_names_dict = model.model.names
    bbox_annotator = spv.BoundingBoxAnnotator(thickness=2, color=class_colors)
    label_annotator = spv.LabelAnnotator(color=class_colors, text_color=spv.Color.from_hex("#000000"))

    # Reading frames from the webcam
    cap = cv2.VideoCapture(args.camera_id)

    # Exporting to GIF
    frames = []
    times = []
    make_gif = args.save_gif is not None
    if make_gif:
        if Path(args.save_gif).is_file():
            path_gif = Path(args.save_gif)
        else:
            path_gif = Path(args.save_gif) / "live_demo.gif"

    # Read from camera and run the YOLO model on each frame
    while True:
        frame_ok, frame = cap.read()

        if frame_ok:
            start_time = time.time()
            result = model(frame, agnostic_nms=True, verbose=False)[0]
            detections = spv.Detections.from_ultralytics(result)

            frame = annotate_frame(frame, detections, bbox_annotator, label_annotator, class_names_dict)
            cv2.imshow("Face parts", frame)
            k = cv2.waitKey(1)

            if make_gif:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                times.append(time.time() - start_time)

            if k == ord("q"):
                break

    cv2.destroyAllWindows()
    cap.release()

    # Exporting to GIF
    # Source: https://pysource.com/2021/03/25/create-an-animated-gif-in-real-time-with-opencv-and-python/
    if make_gif:
        print("\nSaving the stream to ", path_gif)
        avg_time = np.array(times).mean()
        fps = round(1 / avg_time)
        imageio.mimsave(path_gif, frames, format='GIF', fps=fps)
