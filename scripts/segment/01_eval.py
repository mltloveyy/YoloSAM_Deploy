import argparse
import time

import cv2
import numpy as np
from ultralytics.models.sam import SAM2Predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/sam2.1_b.pt", help="The .pt model file for inference")
    parser.add_argument("--image", type=str, default="data/test.jpg", help="The data source for inference")
    parser.add_argument("--imgsz", type=int, default=1024, help="Target image size for inference")
    parser.add_argument("--points", type=int, nargs="+", help="Points prompt")
    parser.add_argument("--boxes", type=int, nargs="+", help="Boxes prompt")
    args = parser.parse_args()

    # args.points = [500, 720, 560, 1070]
    # args.boxes = [430, 551, 664, 1169, 306, 195, 442, 532]

    # prompt check
    num_boxes = len(args.boxes) // 4 if args.boxes is not None else 0
    num_points = len(args.points) // 2 if args.points is not None else 0
    if num_boxes == 0 and num_points == 0:
        parser.error("Either --points(>=2) or --box(>=4) must be provided")

    # init model
    overrides = {"conf": 0.25, "imgsz": args.imgsz, "model": args.model, "save": False, "verbose": False}
    sam = SAM2Predictor(overrides=overrides)

    # Inference
    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    t0 = time.time()
    sam.set_image(img)
    t1 = time.time()
    print(f"Speed: {1000 * (t1 - t0):.1f}ms preprocess+inference at shape {img.shape}")

    img_prompt = img.copy()
    img_result = img.copy()
    # multi boxes prompt
    if num_boxes > 1:
        boxes = np.array(args.boxes[: num_boxes * 4]).reshape(-1, 4)
        for i in range(num_boxes):
            cv2.rectangle(img_prompt, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 255), 2)  # 黄色框
        t2 = time.time()
        results = sam(bboxes=boxes, points=None, labels=None)
        t3 = time.time()
        for pts in results[0].masks.xy:
            pts = pts.astype(np.int32)
            cv2.polylines(img_result, [pts], True, (255, 0, 0), 2)  # 蓝色边界
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框
        print(f"Target: {1000 * ((t3 - t2) / num_boxes):.1f}ms segment mask")
    else:
        # one box or multi points prompt
        points = None
        labels = None
        box = None
        if num_points > 0:
            points = np.array(args.points[: num_points * 2]).reshape(1, -1, 2)
            labels = np.ones((1, num_points), dtype=np.int32)
            for i in range(num_points):
                cv2.circle(img_prompt, (points[0][i][0], points[0][i][1]), 5, (0, 255, 0), -1)  # 绿色点
        if num_boxes == 1:
            box = np.array(args.boxes[:4]).reshape(-1, 4)
            cv2.rectangle(img_prompt, (box[0][0], box[0][1]), (box[0][2], box[0][3]), (0, 255, 255), 2)  # 黄色框
        t2 = time.time()
        results = sam(bboxes=box, points=points, labels=labels)
        t3 = time.time()
        pts = results[0].masks.xy[0].astype(np.int32)
        cv2.drawContours(img_result, [pts], -1, (255, 0, 0), 2)  # 蓝色边界
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框
        print(f"Target: {1000 * (t3 - t2):.1f}ms segment mask")
    cv2.imwrite(f"{args.image[:-4]}_segment.jpg", cv2.hconcat([img_prompt, img_result]))
