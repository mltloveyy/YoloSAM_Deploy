import argparse
import json
import os

import cv2
import numpy as np
from ultralytics import SAM


def save_labelme_json(image_path, shapes, image_height, image_width):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(os.path.dirname(os.path.abspath(image_path)), base_name + ".json")

    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }

    with open(json_path, "w") as f:
        json.dump(labelme_data, f, indent=2)
    print(f"Annotation saved to: {json_path}")


# 交互式SAM分割工具，使用鼠标点击正负点获取分割结果，显示轮廓和矩形，并保存标注文件(coco格式JSON)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_model", type=str, default="models/sam2.1_b.pt", help="The .pt model file for inference")
    parser.add_argument("--image", type=str, default="data/test.jpg", help="The image source for annotation")
    args = parser.parse_args()

    sam = SAM(args.sam_model)

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to load image: {args.image}")
    sam.set_image(img)
    img_h, img_w = img.shape[:2]

    state = {
        "points": [],  # (x, y)
        "labels": [],  # 1=positive, 0=negative
        "results": None,
        "saved_shapes": [],  # {"label": str, "points": [[x,y],...]}
    }

    def draw():
        img_show = img.copy()

        for shape in state["saved_shapes"]:
            pts = np.array(shape["points"], np.int32)
            cv2.polylines(img_show, [pts], True, (255, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(pts)
            cv2.putText(img_show, shape["label"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        for pt, label in zip(state["points"], state["labels"]):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(img_show, pt, 5, color, -1)

        if state["results"] is not None and state["results"][0].masks is not None:
            pts = state["results"][0].masks.xy[0].astype(np.int32)
            cv2.polylines(img_show, [pts], True, [0, 0, 255], 2)
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("SAM Interactive", img_show)

    def run_predict():
        if len(state["points"]) == 0:
            state["results"] = None
            return
        state["results"] = sam(bboxes=None, points=[state["points"]], labels=[state["labels"]])  # [[point1, point2, ...]], [[label1, label2, ...]]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # left: positive
            state["points"].append((x, y))
            state["labels"].append(1)
            run_predict()
            draw()
        elif event == cv2.EVENT_RBUTTONDOWN:  # right: negative
            state["points"].append((x, y))
            state["labels"].append(0)
            run_predict()
            draw()

    cv2.namedWindow("SAM Interactive", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SAM Interactive", on_mouse)
    draw()

    print("=== Instructions ===")
    print("left click  - add positive (green)")
    print("right click - add negative (red)")
    print("press 's'   - save current mask, enter class name")
    print("press 'c'   - clear all points")
    print("press 'ESC' - exit")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("c"):
            state["points"].clear()
            state["labels"].clear()
            state["results"] = None
            draw()
            print("Clear all points")
        elif key == ord("s"):
            if state["results"] is None or state["results"][0].masks is None:
                print("No segmentation result to save. Please add points first.")
                continue

            mask_pts = state["results"][0].masks.xy[0].astype(np.int32)
            if len(mask_pts) < 3:
                print("Invalid mask (too few points).")
                continue

            class_name = input("Enter class name (or press Enter to cancel): ").strip()
            if not class_name:
                print("Empty class name, save cancelled.")
                continue

            shape = {"label": class_name, "points": mask_pts.tolist(), "group_id": None, "shape_type": "polygon", "flags": {}}  # [[x,y], ...]
            state["saved_shapes"].append(shape)

            save_labelme_json(args.image, state["saved_shapes"], img_h, img_w)

            state["points"].clear()
            state["labels"].clear()
            state["results"] = None

            draw()
            print(f"Shape '{class_name}' saved. You can now annotate the next object.")

    cv2.destroyAllWindows()
