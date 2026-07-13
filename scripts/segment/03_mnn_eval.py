import argparse
import time

import MNN
import MNN.cv as cv
import MNN.numpy as np


class MNNPredictor:
    def __init__(self, enc_path, dec_path, thread, precision, backend):
        rt = MNN.nn.create_runtime_manager(({"numThread": thread, "precision": precision, "backend": backend},))
        self.enc = MNN.nn.load_module_from_file(enc_path, [], [], runtime_manager=rt, shape_mutable=False)
        self.dec = MNN.nn.load_module_from_file(dec_path, [], [], runtime_manager=rt)

    def set_image(self, image_path, imgsz):
        # Pre-process
        self.ori_image = cv.imread(image_path)
        self.ih, self.iw, _ = self.ori_image.shape
        self.scale = imgsz / max(self.ih, self.iw)
        self.new_h = int(self.ih * self.scale)
        self.new_w = int(self.iw * self.scale)
        t0 = time.time()
        image = cv.resize(self.ori_image, (self.new_w, self.new_h), 0.0, 0.0, cv.INTER_LINEAR, -1, [123.675, 116.28, 103.53], [1 / 58.395, 1 / 57.12, 1 / 57.375])
        image = np.pad(image, [[0, imgsz - self.new_h], [0, imgsz - self.new_w], [0, 0]], "constant")
        image = np.expand_dims(image, 0)
        image = MNN.expr.convert(image, MNN.expr.NCHW)
        # Inference
        t1 = time.time()
        enc_outputs = self.enc.forward([image])
        self.image_embed = MNN.expr.convert(enc_outputs[0], MNN.expr.NCHW)
        self.high_feats0 = MNN.expr.convert(enc_outputs[1], MNN.expr.NCHW)
        self.high_feats1 = MNN.expr.convert(enc_outputs[2], MNN.expr.NCHW)
        t2 = time.time()
        print(f"Speed: {1000 * (t1 - t0):.1f}ms preprocess, {1000 * (t2 - t1):.1f}ms inference at shape {image.shape}")

    def prompt_inference(self, box=None, points=None):
        coords, labels = None, None
        if box is not None:
            coords = box * self.scale
            labels = np.array([[2, 3]], dtype=np.float32)
        if points is not None:
            point_coords = points * self.scale
            point_labels = np.ones((1, points.shape[1]), dtype=np.float32)
            coords = point_coords if coords is None else np.concatenate([coords, point_coords], axis=1)
            labels = point_labels if labels is None else np.concatenate([labels, point_labels], axis=1)
        t3 = time.time()
        dec_outputs = self.dec.onForward([coords, labels, self.image_embed, self.high_feats0, self.high_feats1])
        mask = MNN.expr.convert(dec_outputs[0], MNN.expr.NCHW)
        # Post-process
        mask = np.squeeze(mask)[: self.new_h, : self.new_w] > 0
        mask = mask.astype(np.uint8)
        mask = cv.resize(mask, (self.iw, self.ih), interpolation=cv.INTER_NEAREST)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        t4 = time.time()
        print(f"Target: {1000 * (t4 - t3):.1f}ms segment mask")
        return contours

    def __call__(self, box, points):
        return self.prompt_inference(box, points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc", type=str, default="models/sam2.1_b_enc.mnn", help="The .mnn model file for image encoder")
    parser.add_argument("--dec", type=str, default="models/sam2.1_b_dec.mnn", help="The .mnn model file for prompt decoder")
    parser.add_argument("--image", type=str, default="data/test.jpg", help="The data source for inference")
    parser.add_argument("--imgsz", type=int, default=1024, help="Target image size for inference")
    parser.add_argument("--points", type=int, nargs="+", help="Points prompt")
    parser.add_argument("--boxes", type=int, nargs="+", help="Boxes prompt")
    parser.add_argument("--thread", type=int, default=2, help="inference using thread")
    parser.add_argument("--precision", type=int, default=0, help="inference precision: 0(normal), high, low, low_bf16")
    parser.add_argument("--backend", type=int, default=0, help="inference backend: 0(CPU), METAL, CUDA, OPENCL, AUTO, NN, OPENGL, VULKAN")
    args = parser.parse_args()

    # args.points = [500, 720, 560, 1070]
    # args.boxes = [430, 551, 664, 1169 , 306, 195, 442, 532]

    # prompt check
    num_boxes = len(args.boxes) // 4 if args.boxes is not None else 0
    num_points = len(args.points) // 2 if args.points is not None else 0
    if num_boxes == 0 and num_points == 0:
        parser.error("Either --points or --box must be provided")

    # init models
    sam = MNNPredictor(args.enc, args.dec, args.thread, args.precision, args.backend)

    # inference
    sam.set_image(args.image, args.imgsz)

    img_prompt = sam.ori_image.copy()
    img_result = sam.ori_image.copy()
    # multi boxes prompt
    if num_boxes > 1:
        boxes = np.array(args.boxes[: num_boxes * 4]).reshape(-1, 2, 2)
        for i in range(num_boxes):
            cv.rectangle(img_prompt, (boxes[i, 0, 0], boxes[i, 0, 1]), (boxes[i, 1, 0], boxes[i, 1, 1]), (0, 255, 255), 2)  # 黄色框
            contours = sam(box=boxes[i : i + 1, :, :], points=None)
            cv.drawContours(img_result, contours, 0, (255, 0, 0), 2)  # 蓝色边界
            x, y, w, h = cv.boundingRect(contours[0])
            cv.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框
    else:
        # one box or multi points prompt
        box, points = None, None
        if num_boxes > 0:
            box = np.array(args.boxes[:4]).reshape(-1, 2, 2)
            cv.rectangle(img_prompt, (box[0, 0, 0], box[0, 0, 1]), (box[0, 1, 0], box[0, 1, 1]), (0, 255, 255), 2)  # 黄色框
        if num_points > 0:
            points = np.array(args.points[: num_points * 2]).reshape(1, -1, 2)
            for i in range(num_points):
                cv.circle(img_prompt, (points[0][i][0], points[0][i][1]), 5, (0, 255, 0), -1)  # 绿色点
        contours = sam(box=box, points=points)
        cv.drawContours(img_result, contours, 0, (255, 0, 0), 2)  # 蓝色边界
        x, y, w, h = cv.boundingRect(contours[0])
        cv.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框
    cv.imwrite(f"{args.image[:-4]}_segment_mnn.jpg", cv.hconcat([img_prompt, img_result]))
