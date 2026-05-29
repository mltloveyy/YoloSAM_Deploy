import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/yolo26x_best.pt", help="The .pt model file for inference")
    parser.add_argument("--input", type=str, default="data", help="The data source for inference")
    parser.add_argument("--imgsz", type=int, default=640, help="Target image size for inference")
    args = parser.parse_args()

    # Inference
    model = YOLO(args.model)
    results = model(source=args.input, imgsz=args.imgsz, conf=0.25, iou=0.8, stream=True)

    # Visualization
    output_dir = os.path.join(args.input, "pt_results")
    for r in results:
        save_subdir = os.path.join(output_dir, Path(r.path).stem)
        os.makedirs(save_subdir, exist_ok=True)
        r.save(filename=os.path.join(save_subdir, "0result.jpg"), conf=False)
        boxes = r.boxes
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box.xyxy[0]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            patch = r.orig_img[y0:y1, x0:x1]
            name = r.names[int(box.cls)]
            cv2.imwrite(os.path.join(save_subdir, f"{name}_{i}.jpg"), patch)
