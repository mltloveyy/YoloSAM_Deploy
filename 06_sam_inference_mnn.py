import argparse

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

    args.points = [500, 720, 560, 1070]
    # args.boxes = [430, 551, 664, 1169]

    # prompt check
    num_boxes = len(args.boxes) // 4 if args.boxes is not None else 0
    num_points = len(args.points) // 2 if args.points is not None else 0
    if num_boxes == 0 and num_points == 0:
        parser.error("Either --points or --box must be provided")

    # init models
    config = {}
    config["precision"] = args.precision
    config["backend"] = args.backend
    config["numThread"] = args.thread
    rt = MNN.nn.create_runtime_manager((config,))
    enc = MNN.nn.load_module_from_file(args.enc, ["image"], ["image_embed", "high_res_feats_0", "high_res_feats_1"], runtime_manager=rt)
    dec = MNN.nn.load_module_from_file(
        args.dec,
        ["point_coords", "point_labels", "image_embed", "high_res_feats_0", "high_res_feats_1"],
        ["mask"],
        runtime_manager=rt,
    )

    # preprocess
    image = cv.imread(args.image)
    ih, iw, _ = image.shape
    scale = args.imgsz / max(ih, iw)
    new_h, new_w = int(ih * scale), int(iw * scale)
    image = cv.resize(image, (new_w, new_h), 0.0, 0.0, cv.INTER_LINEAR, -1, [123.675, 116.28, 103.53], [1 / 58.395, 1 / 57.12, 1 / 57.375])
    image = np.pad(image, [[0, args.imgsz - new_h], [0, args.imgsz - new_w], [0, 0]], "constant")
    # encode
    enc_inputs = np.expand_dims(image, 0)
    enc_inputs = MNN.expr.convert(enc_inputs, MNN.expr.NCHW)
    enc_outputs = enc.forward([enc_inputs])
    image_embed = enc_outputs[0]
    high_feats0 = enc_outputs[1]
    high_feats1 = enc_outputs[2]

    img_prompt = image.copy()
    img_result = image.copy()
    if num_boxes > 1:
        boxes = np.array(args.boxes[: num_boxes * 4]).reshape(-1, 2, 2)
        for i in range(num_boxes):
            cv.rectangle(img_prompt, (boxes[i, 0, 0], boxes[i, 0, 1]), (boxes[i, 1, 0], boxes[i, 1, 1]), (0, 255, 255), 2)  # 黄色框
            box_coords = boxes[i : i + 1, :, :] * scale
            box_labels = np.array([[2, 3]])
            dec_inputs = [box_coords, box_labels, image_embed, high_feats0, high_feats1]
            dec_outputs = dec.forward(dec_inputs)
            mask = dec_outputs[0]
            # postprocess
            mask = cv.resize(np.squeeze(mask)[:new_h, :new_w], (iw, ih), interpolation=cv.INTER_NEAREST)
            mask = ((mask > 0) * 255).astype(np.uint8)
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.polylines(img_result, contours, True, (255, 0, 0), 2)  # 蓝色边界
            x, y, w, h = cv.boundingRect(contours[0])
            cv.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框
    else:
        coords, labels = None, None
        if num_boxes == 1:
            box = np.array(args.boxes[:4]).reshape(-1, 2, 2)
            cv.rectangle(img_prompt, (box[0, 0, 0], box[0, 0, 1]), (box[0, 1, 0], box[0, 1, 1]), (0, 255, 255), 2)  # 黄色框
            coords = box * scale
            labels = np.array([[2, 3]])
        if num_points > 0:
            points = np.array(args.points[: num_points * 2]).reshape(1, -1, 2)
            for i in range(num_points):
                cv.circle(img_prompt, (points[0][i][0], points[0][i][1]), 5, (0, 255, 0), -1)  # 绿色点
            point_coords = points * scale
            point_labels = np.ones((1, num_points))
            coords = point_coords if coords is None else np.concatenate([coords, point_coords], axis=1)
            labels = point_labels if labels is None else np.concatenate([labels, point_labels], axis=1)
        dec_inputs = [coords, labels, image_embed, high_feats0, high_feats1]
        dec_outputs = dec.forward(dec_inputs)
        mask = dec_outputs[0]
        # postprocess
        mask = cv.resize(np.squeeze(mask)[:new_h, :new_w], (iw, ih), interpolation=cv.INTER_NEAREST)
        mask = ((mask > 0) * 255).astype(np.uint8)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.polylines(img_result, contours, True, (255, 0, 0), 2)  # 蓝色边界
        x, y, w, h = cv.boundingRect(contours[0])
        cv.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框
    cv.imwrite(f"{args.image[:-4]}_segment.jpg", cv.hconcat([img_prompt, img_result]))
