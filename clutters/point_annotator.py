import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

img_path = "/home/yy/workspace/datasets/railwaytools/val/20241115191916.jpg"

# Create SAMPredictor
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    imgsz=1024,
    model="Yolo/models/sam_b.pt",
)
predictor = SAMPredictor(overrides=overrides)

img_show = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
predictor.set_image(img_show)

points = [[[300, 430], [700, 430], [500, 1000]]]  # [w, h]
labels = [[1, 1, 1]]
for i in range(len(points[0])):
    cv2.circle(img_show, (points[0][i][0], points[0][i][1]), 5, (0, 255, 0), -1)

results = predictor(bboxes=None, points=points, labels=labels)

mask_pts = results[0].masks.xy[0].astype(np.int32)

cv2.polylines(img_show, [mask_pts], True, [0, 0, 255], 2)
# cv2.fillPoly(img_show, [mask_pts], [0, 0, 255])

box = cv2.boundingRect(mask_pts)
cv2.rectangle(img_show, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

cv2.imshow("img", img_show)
cv2.waitKey(0)
