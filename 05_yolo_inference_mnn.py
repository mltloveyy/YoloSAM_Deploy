import argparse
import os
import time
from glob import glob
from pathlib import Path

import cv2
import MNN
import MNN.cv as cv
import MNN.numpy as np
import yaml


def inference(net, image_path, imgsz):
    # Pre-process
    image = cv.imread(image_path)
    ih, iw, _ = image.shape
    length = max((ih, iw))
    scale = length / imgsz
    image = np.pad(image, [[0, length - ih], [0, length - iw], [0, 0]], "constant")
    image = cv.resize(image, (imgsz, imgsz), 0.0, 0.0, cv.INTER_LINEAR, -1, [0.0, 0.0, 0.0], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])
    infer_size = f"{int(ih/scale)}x{int(iw/scale)}"

    # Inference
    input_var = np.expand_dims(image, 0)
    input_var = MNN.expr.convert(input_var, MNN.expr.NCHW)
    t0 = time.time()
    output_var = net.forward(input_var)
    latency = (time.time() - t0) * 1000
    output_var = MNN.expr.convert(output_var, MNN.expr.NCHW)  # [1, 300, 6]; 6 means: [x0, y0, x1, y1, scores, class_id]
    output_var = output_var.squeeze().transpose()

    # Post-process
    x0 = output_var[0]
    y0 = output_var[1]
    x1 = output_var[2]
    y1 = output_var[3]
    scores = output_var[4]
    class_ids = output_var[5]
    boxes = np.stack([x0, y0, x1, y1], axis=1)
    result_ids = MNN.expr.nms(boxes, scores, 100, 0.8, 0.25)  # nms
    result_boxes = boxes[result_ids] * scale
    result_scores = scores[result_ids]
    result_class_ids = class_ids[result_ids]
    return result_boxes, result_class_ids, result_scores, latency, infer_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/yolo26x.mnn", help="The .mnn model file for inference")
    parser.add_argument("--input", type=str, default="data", help="The data source for inference")
    parser.add_argument("--config", type=str, default="data/config.yaml", help="The configuration file")
    parser.add_argument("--imgsz", type=int, default=640, help="Target image size for inference")
    parser.add_argument("--precision", type=str, default="normal", help="inference precision: normal, low, high, lowBF")
    parser.add_argument("--backend", type=str, default="CPU", help="inference backend: CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI")
    parser.add_argument("--thread", type=int, default=4, help="inference using thread: int")
    args = parser.parse_args()

    # Collect all image paths
    exts = ["jpeg", "jpg", "png"]
    input_list = []
    for ext in exts:
        input_list.extend(glob(os.path.join(args.input, f"*.{ext}")))

    # Load names from config file
    with open(args.config, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names_dict = data["names"]

    # Load model
    config = {}
    config["precision"] = args.precision
    config["backend"] = args.backend
    config["numThread"] = args.thread
    rt = MNN.nn.create_runtime_manager((config,))
    net = MNN.nn.load_module_from_file(args.model, [], [], runtime_manager=rt, shape_mutable=False)

    # Warmup infernece
    dummy_input = np.zeros((1, 3, args.imgsz, args.imgsz))
    warmup_iters = 3
    for i in range(warmup_iters):
        _ = net.forward(dummy_input)

    output_dir = os.path.join(args.input, "mnn_results")
    for i, image_path in enumerate(input_list):
        # Inference
        boxes, class_ids, scores, latency, infer_size = inference(net, image_path, args.imgsz)

        # Visualization
        save_subdir = os.path.join(output_dir, Path(image_path).stem)
        os.makedirs(save_subdir, exist_ok=True)
        image = cv2.imread(image_path)
        img_show = image.copy()
        name_counts = {}
        for j in range(len(boxes)):
            name = names_dict.get(int(class_ids[j]), "unknown")
            x0, y0, x1, y1 = boxes[j].read_as_tuple()
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            patch = image[y0:y1, x0:x1]
            cv2.imwrite(os.path.join(save_subdir, f"{name}_{j}.jpg"), patch)
            cv2.putText(img_show, f"{name}: {scores[j]:.3f}", (x0, max(y0 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(img_show, (x0, y0), (x1, y1), (0, 0, 255), 2)
            name_counts[name] = name_counts.get(name, 0) + 1
        cv2.imwrite(os.path.join(save_subdir, "0result.jpg"), img_show)
        print(f"image {i + 1}/{len(input_list)} {image_path}: {infer_size} {", ".join([f"{v} {k}" for k, v in name_counts.items()])} {latency:.1f}ms")
