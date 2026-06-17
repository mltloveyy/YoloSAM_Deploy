import time

import cv2
import numpy as np
import onnxruntime as ort

enc_session = ort.InferenceSession("models/sam2.1_b_enc.onnx")
dec_session = ort.InferenceSession("models/sam2.1_b_dec.onnx")
imgsz = 1024
mean = np.array([123.675, 116.28, 103.53])
std = np.array([58.395, 57.12, 57.375])

ori_img = cv2.imread("data/test.jpg")

# preprocess
ih, iw = ori_img.shape[:2]
scale = imgsz / max(ih, iw)
new_w, new_h = int(iw * scale), int(ih * scale)
image = cv2.resize(ori_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
image = (image - mean) / std
image = np.pad(image, [[0, imgsz - new_h], [0, imgsz - new_w], [0, 0]], "constant")
image = np.transpose(image.astype(np.float32), (2, 0, 1))  # HWC -> CHW
image = np.expand_dims(image, axis=0)

image_embed, high_feats0, high_feats1 = enc_session.run(["image_embed", "high_res_feats_0", "high_res_feats_1"], {"image": image})

points_list = [[[430, 551], [664, 1169], [500, 720], [560, 1070]]]
labels_list = [[2, 3, 1, 1]]
points = np.array(points_list, dtype=np.float32) * scale
labels = np.array(labels_list, dtype=np.int64)

input_dict = {
    "point_coords": points,
    "point_labels": labels,
    "image_embed": image_embed,
    "high_res_feats_0": high_feats0,
    "high_res_feats_1": high_feats1,
}
for _ in range(5):
    _ = dec_session.run(["mask"], input_dict)
t0 = time.time()
for _ in range(20):
    _ = dec_session.run(["mask"], input_dict)
t1 = time.time()
print(f"decoder cost: {(50*(t1 - t0)):.1f}ms")
mask = dec_session.run(["mask"], input_dict)
# postprocess
mask = cv2.resize(np.squeeze(mask)[:new_h, :new_w], (iw, ih), interpolation=cv2.INTER_NEAREST)
mask = ((mask > 0) * 255).astype(np.uint8)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.polylines(ori_img, contours, True, (255, 0, 0), 2)
x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(ori_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
for p in points_list[0]:
    cv2.circle(ori_img, p, 3, (0, 0, 255), 5)
cv2.imwrite("result.jpg", ori_img)
