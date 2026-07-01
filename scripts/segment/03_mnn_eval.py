import argparse
import time

import MNN
import MNN.cv as cv
import MNN.numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc", type=str, default="models/sam2.1_b_enc.mnn", help="The .mnn model file for image encoder")
    parser.add_argument("--dec", type=str, default="models/sam2.1_b_dec.mnn", help="The .mnn model file for prompt decoder")
    parser.add_argument("--image", type=str, default="data/test.jpg", help="The data source for inference")
    parser.add_argument("--imgsz", type=int, default=1024, help="Target image size for inference")
    parser.add_argument("--points", type=int, nargs="+", help="Points prompt")
    parser.add_argument("--boxes", type=int, nargs="+", help="Boxes prompt")
    parser.add_argument("--precision", type=str, default="normal", help="inference precision: normal, low, high, lowBF")
    parser.add_argument("--backend", type=str, default="CPU", help="inference backend: CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI")
    parser.add_argument("--thread", type=int, default=4, help="inference using thread: int")
    args = parser.parse_args()

    # args.points = [500, 720, 560, 1070]
    # args.boxes = [430, 551, 664, 1169 , 306, 195, 442, 532]

    # prompt check
    num_boxes = len(args.boxes) // 4 if args.boxes is not None else 0
    num_points = len(args.points) // 2 if args.points is not None else 0
    if num_boxes == 0 and num_points == 0:
        parser.error("Either --points or --box must be provided")

    # init models
    rt = MNN.nn.create_runtime_manager(({"precision": args.precision, "backend": args.backend, "numThread": args.thread},))
    enc = MNN.nn.load_module_from_file(args.enc, [], [], runtime_manager=rt)
    dec = MNN.nn.load_module_from_file(args.dec, [], [], runtime_manager=rt)

    # Pre-process
    ori_image = cv.imread(args.image)
    ih, iw, _ = ori_image.shape
    scale = args.imgsz / max(ih, iw)
    new_h, new_w = int(ih * scale), int(iw * scale)
    t0 = time.time()
    image = cv.resize(ori_image, (new_w, new_h), 0.0, 0.0, cv.INTER_LINEAR, -1, [123.675, 116.28, 103.53], [1 / 58.395, 1 / 57.12, 1 / 57.375])
    image = np.pad(image, [[0, args.imgsz - new_h], [0, args.imgsz - new_w], [0, 0]], "constant")
    image = np.expand_dims(image, 0)
    image = MNN.expr.convert(image, MNN.expr.NCHW)
    # Inference
    t1 = time.time()
    enc_outputs = enc.forward([image])
    image_embed = MNN.expr.convert(enc_outputs[0], MNN.expr.NCHW)
    high_feats0 = MNN.expr.convert(enc_outputs[1], MNN.expr.NCHW)
    high_feats1 = MNN.expr.convert(enc_outputs[2], MNN.expr.NCHW)
    t2 = time.time()
    print(f"Speed: {1000 * (t1 - t0):.1f}ms preprocess, {1000 * (t2 - t1):.1f}ms inference at shape {image.shape}")

    img_prompt = ori_image.copy()
    img_result = ori_image.copy()
    # multi boxes prompt
    if num_boxes > 1:
        boxes = np.array(args.boxes[: num_boxes * 4]).reshape(-1, 2, 2)
        for i in range(num_boxes):
            cv.rectangle(img_prompt, (boxes[i, 0, 0], boxes[i, 0, 1]), (boxes[i, 1, 0], boxes[i, 1, 1]), (0, 255, 255), 2)  # 黄色框
            box_coords = boxes[i : i + 1, :, :] * scale
            box_labels = np.array([[2, 3]], dtype=np.float32)
            t3 = time.time()
            dec_outputs = dec.onForward([box_coords, box_labels, image_embed, high_feats0, high_feats1])
            mask = MNN.expr.convert(dec_outputs[0], MNN.expr.NCHW)
            # Post-process
            t4 = time.time()
            mask = np.squeeze(mask)[:new_h, :new_w] > 0
            mask = mask.astype(np.uint8)
            mask = cv.resize(mask, (iw, ih), interpolation=cv.INTER_NEAREST)
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            t5 = time.time()
            cv.drawContours(img_result, contours, -1, (255, 0, 0), 2)  # 蓝色边界
            x, y, w, h = cv.boundingRect(contours[0])
            cv.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框
            print(f"Target{i+1}: {1000 * (t4 - t3):.1f}ms decode mask, {1000 * (t5 - t4):.1f}ms postprocess")
    else:
        # one box or multi points prompt
        coords, labels = None, None
        if num_boxes > 0:
            box = np.array(args.boxes[:4]).reshape(-1, 2, 2)
            cv.rectangle(img_prompt, (box[0, 0, 0], box[0, 0, 1]), (box[0, 1, 0], box[0, 1, 1]), (0, 255, 255), 2)  # 黄色框
            coords = box * scale
            labels = np.array([[2, 3]], dtype=np.float32)
        if num_points > 0:
            points = np.array(args.points[: num_points * 2]).reshape(1, -1, 2)
            for i in range(num_points):
                cv.circle(img_prompt, (points[0][i][0], points[0][i][1]), 5, (0, 255, 0), -1)  # 绿色点
            point_coords = points * scale
            point_labels = np.ones((1, num_points), dtype=np.float32)
            coords = point_coords if coords is None else np.concatenate([coords, point_coords], axis=1)
            labels = point_labels if labels is None else np.concatenate([labels, point_labels], axis=1)
        t3 = time.time()
        dec_outputs = dec.onForward([coords, labels, image_embed, high_feats0, high_feats1])
        mask = MNN.expr.convert(dec_outputs[0], MNN.expr.NCHW)
        # Post-process
        t4 = time.time()
        mask = np.squeeze(mask)[:new_h, :new_w] > 0
        mask = mask.astype(np.uint8)
        mask = cv.resize(mask, (iw, ih), interpolation=cv.INTER_NEAREST)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        t5 = time.time()
        cv.drawContours(img_result, contours, -1, (255, 0, 0), 2)  # 蓝色边界
        x, y, w, h = cv.boundingRect(contours[0])
        cv.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框
        print(f"Target: {1000 * (t4 - t3):.1f}ms decode mask, {1000 * (t5 - t4):.1f}ms postprocess")
    cv.imwrite(f"{args.image[:-4]}_segment_mnn.jpg", cv.hconcat([img_prompt, img_result]))
