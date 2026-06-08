import argparse
import json
import os

import cv2
import numpy as np
from ultralytics.models.sam import SAM2Predictor


def load_labelme_bboxes(json_path):
    """加载 labelme JSON 文件，提取所有矩形标注的 label 和 bbox 两点。
    返回列表，每个元素为 {'label': str, 'points': [[x1,y1],[x2,y2]]}"""
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    shapes = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "rectangle":
            shapes.append({"label": shape["label"], "points": shape["points"]})  # [[x1,y1],[x2,y2]]
    return shapes


def save_labelme_json(image_path, loaded_shapes, img_h, img_w):
    """将 loaded_shapes 保存为 labelme JSON 文件。"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(os.path.dirname(os.path.abspath(image_path)), base_name + ".json")

    labelme_shapes = []
    for s in loaded_shapes:
        labelme_shapes.append({"label": s["label"], "points": s["points"], "shape_type": "rectangle", "group_id": None, "flags": {}})

    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": labelme_shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }
    with open(json_path, "w") as f:
        json.dump(labelme_data, f, indent=2)
    print(f"Annotation saved to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/sam2.1_b.pt")
    parser.add_argument("--image", type=str, default="data/test.jpg")
    args = parser.parse_args()

    overrides = {"conf": 0.25, "imgsz": 1024, "model": args.model, "save": False}
    sam = SAM2Predictor(overrides=overrides)

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to load image: {args.image}")
    sam.set_image(img)
    img_h, img_w = img.shape[:2]

    json_path = os.path.splitext(args.image)[0] + ".json"
    loaded_shapes = load_labelme_bboxes(json_path)  # 原始 bbox 列表

    state = {
        "predicted_shapes": [],  # bbox 推理出的多边形列表，每个元素 {"label": str, "points": [[x,y],...]}
        "selected_idx": None,  # 当前选中的 predicted_shapes 索引，None 表示未选中
        "points": [],  # 点提示：当前积累的点坐标和标签
        "labels": [],
        "point_type": 1,  # 当前添加点的类型，1=positive, 0=negative
        "temp_result": None,  # 点提示的临时推理结果
    }

    def run_point_predict():
        """根据当前累积的点运行 SAM 点提示推理，更新 temp_result"""
        if len(state["points"]) == 0:
            state["temp_result"] = None
            return
        state["temp_result"] = sam(bboxes=None, points=[state["points"]], labels=[state["labels"]])

    def bbox_to_xyxy(pts):
        """将矩形两点 [[x1,y1],[x2,y2]] 转换为 [xmin, ymin, xmax, ymax]"""
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        return [xmin, ymin, xmax, ymax]

    def draw():
        img_show = img.copy()

        # 有 predicted_shapes 时，绘制多边形与标签，不再绘制 bbox
        if state["predicted_shapes"]:
            for i, shape in enumerate(state["predicted_shapes"]):
                pts = np.array(shape["points"], np.int32)
                if len(pts) < 3:
                    continue
                # 默认红色轮廓，选中的用黄色高亮
                color = (0, 255, 255) if i == state["selected_idx"] else (0, 0, 255)
                cv2.polylines(img_show, [pts], True, color, 2)
                # 标签放在多边形第一个点附近
                cv2.putText(img_show, shape["label"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # 未做 bbox 推理时，绘制原始 bbox 矩形和标签
            for shape in loaded_shapes:
                pts = np.array(shape["points"], np.int32).reshape(-1, 2)
                cv2.rectangle(img_show, pts[0], pts[1], (0, 255, 255), 2)
                cv2.putText(img_show, shape["label"], (pts[0][0], pts[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 绘制点提示的点（绿色正，红色负）
        for pt, lbl in zip(state["points"], state["labels"]):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            print(f"Prompt points: {pt}, label: {lbl}")
            cv2.circle(img_show, pt, 5, color, -1)

        # 绘制点提示的临时分割结果
        if state["temp_result"] is not None and state["temp_result"][0].masks is not None:
            pts = state["temp_result"][0].masks.xy[0].astype(np.int32)
            cv2.polylines(img_show, [pts], True, (255, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 左上角：若选中形状则显示其 label
        if state["selected_idx"] is not None and state["predicted_shapes"]:
            label = state["predicted_shapes"][state["selected_idx"]]["label"]
            cv2.putText(img_show, f"Selected: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # 右上角：显示当前点的类型
        mode_text = "Mode: Pos" if state["point_type"] == 1 else "Mode: Neg"
        mode_color = (0, 255, 0) if state["point_type"] == 1 else (0, 0, 255)
        cv2.putText(img_show, mode_text, (img_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

        cv2.imshow("SAM Interactive", img_show)

    # ---------- 鼠标回调 ----------
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 情况 1：已存在 bbox 推理的多边形
        if state["predicted_shapes"]:
            hit = False
            for i, shape in enumerate(state["predicted_shapes"]):
                pts = np.array(shape["points"], np.int32)
                if len(pts) < 3:
                    continue
                if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                    # 点击在形状内部 → 选中该形状
                    state["selected_idx"] = i
                    state["points"].clear()
                    state["labels"].clear()
                    state["temp_result"] = None
                    hit = True
                    break
            if not hit:
                # 点击在空白区域 → 开始新的点提示分割
                if len(state["points"]) == 0 and state["point_type"] == 0:
                    print("First point must be positive (green). Ignored.")
                    return
                state["selected_idx"] = None
                state["points"].append((x, y))
                state["labels"].append(state["point_type"])
                run_point_predict()
        else:
            # 情况 2：无 bbox 多边形，直接进行点提示累积
            if len(state["points"]) == 0 and state["point_type"] == 0:
                print("First point must be positive (green). Ignored.")
                return
            state["selected_idx"] = None
            state["points"].append((x, y))
            state["labels"].append(state["point_type"])
            run_point_predict()

        draw()

    cv2.namedWindow("SAM Interactive", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SAM Interactive", on_mouse)
    draw()

    print("=== Instructions ===")
    print("left click  - add point / select shape")
    print("right click - (disabled, use 'n' to toggle pos/neg)")
    print("'r'         - run bbox prompt on all loaded bboxes")
    print("'d'         - delete selected shape (only display)")
    print("'s'         - delete selected shape from JSON (if selected) / save point result to JSON")
    print("'n'         - toggle point type (pos/neg)")
    print("backspace   - remove last point")
    print("'c'         - clear all points")
    print("'ESC'       - exit")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        elif key == ord("r"):
            # 按 r：使用 loaded_shapes 的所有 bbox 运行分割
            if not loaded_shapes:
                print("No bbox loaded from JSON.")
            else:
                predicted = []
                for shape in loaded_shapes:
                    bbox_xyxy = bbox_to_xyxy(shape["points"])
                    res = sam(bboxes=[bbox_xyxy])
                    if res[0].masks is not None:
                        mask_pts = res[0].masks.xy[0].astype(np.int32)
                    else:
                        mask_pts = np.empty((0, 2), dtype=np.int32)
                    predicted.append({"label": shape["label"], "points": mask_pts.tolist() if len(mask_pts) > 0 else []})
                state["predicted_shapes"] = predicted
                state["selected_idx"] = None
                state["points"].clear()
                state["labels"].clear()
                state["temp_result"] = None
                print("Bbox prompt finished. Displaying predicted shapes.")
            draw()

        elif key == ord("d"):
            # 删除当前选中的多边形（仅从 predicted_shapes 中移除）
            if state["selected_idx"] is not None and 0 <= state["selected_idx"] < len(state["predicted_shapes"]):
                print(f"Deleted shape '{state['predicted_shapes'][state['selected_idx']]['label']}' from display.")
                del state["predicted_shapes"][state["selected_idx"]]
                state["selected_idx"] = None
                draw()

        elif key == ord("s"):
            if state["selected_idx"] is not None and state["predicted_shapes"]:
                # 有选中形状 → 从 loaded_shapes 和 predicted_shapes 中删除，并更新 JSON
                idx = state["selected_idx"]
                if idx < len(loaded_shapes):
                    removed = loaded_shapes.pop(idx)
                    save_labelme_json(args.image, loaded_shapes, img_h, img_w)
                    print(f"Removed bbox '{removed['label']}' from JSON.")
                # 同步移除对应的 predicted 项
                if idx < len(state["predicted_shapes"]):
                    del state["predicted_shapes"][idx]
                state["selected_idx"] = None
                draw()

            elif state["temp_result"] is not None and state["temp_result"][0].masks is not None:
                # 没有选中形状，但有临时点分割结果 → 保存为矩形 bbox
                mask_pts = state["temp_result"][0].masks.xy[0].astype(np.int32)
                if len(mask_pts) >= 3:
                    class_name = input("Enter class name: ").strip()
                    if class_name:
                        x, y, w, h = cv2.boundingRect(mask_pts)
                        rect_points = [[x, y], [x + w, y + h]]
                        loaded_shapes.append({"label": class_name, "points": rect_points})
                        save_labelme_json(args.image, loaded_shapes, img_h, img_w)
                        state["points"].clear()
                        state["labels"].clear()
                        state["temp_result"] = None
                        state["predicted_shapes"] = []  # 清空旧多边形，可重新按 r
                        print(f"Saved '{class_name}' to JSON.")
                    else:
                        print("Save cancelled.")
                else:
                    print("Invalid mask (too few points).")
            else:
                print("Nothing to save.")
            draw()

        elif key == ord("n"):
            # 切换正负点类型
            state["point_type"] = 1 if state["point_type"] == 0 else 0
            print(f"Point type switched to {'POS' if state['point_type'] == 1 else 'NEG'}.")
            draw()

        elif key == 8:  # backspace
            if state["points"]:
                state["points"].pop()
                state["labels"].pop()
                run_point_predict()
                draw()
                print("Removed last point.")

        elif key == ord("c"):
            state["points"].clear()
            state["labels"].clear()
            state["temp_result"] = None
            draw()
            print("Cleared all points.")

    cv2.destroyAllWindows()
