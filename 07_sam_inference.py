import argparse

import cv2
import numpy as np
from ultralytics.models.sam import SAM2Predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/sam2.1_b.pt", help="The .pt model file for inference")
    parser.add_argument("--image", type=str, default="data/test.jpg", help="The data source for inference")
    parser.add_argument("--points", type=int, nargs="+", help="Points prompt")
    parser.add_argument("--box", type=int, nargs="+", help="Box prompt")
    args = parser.parse_args()

    # args.points = [500, 720, 560, 1070]
    # args.box = [430, 551, 664, 1169]

    # prompt check
    if args.points is None and args.box is None:
        parser.error("Either --points or --box must be provided")
    if args.points is not None and len(args.points) < 2:
        parser.error(f"--points requires at least 2 values, got {len(args.points)}")
    if args.box is not None and len(args.box) != 4:
        parser.error(f"--box requires exactly 4 values (x1 y1 x2 y2), got {len(args.box)}")

    overrides = {"conf": 0.25, "imgsz": 1024, "model": args.model, "save": False}
    sam = SAM2Predictor(overrides=overrides)

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    sam.set_image(img)
    img_h, img_w = img.shape[:2]

    points = None
    labels = None
    box = None
    img_show = img.copy()

    if args.points is not None:
        num_points = len(args.points) // 2
        points = np.array(args.points[: num_points * 2]).reshape(1, -1, 2).tolist()
        labels = np.ones((1, num_points), dtype=np.int32).tolist()
        for i in range(num_points):
            cv2.circle(img_show, (points[0][i][0], points[0][i][1]), 5, (0, 255, 0), -1)  # 绿色点
    if args.box is not None:
        box = [args.box]
        cv2.rectangle(img_show, (args.box[0], args.box[1]), (args.box[2], args.box[3]), (0, 255, 255), 2)  # 黄色框
    cv2.imwrite("prompt.jpg", img_show)

    results = sam(bboxes=box, points=points, labels=labels)

    img_show = img.copy()
    pts = results[0].masks.xy[0].astype(np.int32)
    cv2.polylines(img_show, [pts], True, (255, 0, 0), 2)  # 蓝色边界
    x, y, w, h = cv2.boundingRect(pts)
    cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框

    cv2.imwrite("sam.jpg", img_show)
