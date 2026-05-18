## 预训练模型

### YOLO11

| Model                                                                                | size (pixels) | params (M) | FLOPs (B) | Speed ONNX @CPU (ms) | mAP 50-95 |
| ------------------------------------------------------------------------------------ | ------------- | ---------- | --------- | -------------------- | --------- |
| [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt) | 640           | 2.6        | 6.5       | 56.1 ± 0.8           | 39.5      |
| [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt) | 640           | 9.4        | 21.5      | 90.0 ± 1.2           | 47.0      |
| [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt) | 640           | 20.1       | 68.0      | 183.2 ± 2.0          | 51.5      |
| [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l.pt) | 640           | 25.3       | 86.9      | 238.6 ± 1.4          | 53.4      |
| [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt) | 640           | 56.9       | 194.9     | 462.8 ± 6.7          | 54.7      |

### YOLO26

| Model                                                                                | size (pixels) | params (M) | FLOPs (B) | Speed ONNX @CPU (ms) | mAP 50-95 / (e2e) |
| ------------------------------------------------------------------------------------ | ------------- | ---------- | --------- | -------------------- | ----------------- |
| [YOLO26n](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt) | 640           | 2.4        | 5.4       | 38.9 ± 0.7           | 40.9 / 40.1       |
| [YOLO26s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt) | 640           | 9.5        | 20.7      | 87.2 ± 0.9           | 48.6 / 47.8       |
| [YOLO26m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt) | 640           | 20.4       | 68.2      | 220.0 ± 1.4          | 53.1 / 52.5       |
| [YOLO26l](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt) | 640           | 24.8       | 86.4      | 286.2 ± 2.0          | 55.0 / 54.4       |
| [YOLO26x](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt) | 640           | 55.7       | 193.9     | 525.8 ± 4.0          | 57.5 / 56.9       |

### SAM

| Model                                                                                      | size (pixels) | params (M) | FLOPs (B) |
| ------------------------------------------------------------------------------------------ | ------------- | ---------- | --------- |
| [SAM_base](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam_b.pt)        | 1024          | 90         | 454       |
| [SAM_large](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam_l.pt)       | 1024          | 308        | 1524      |
| [SAM2.1_tiny](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt)  | 1024          | 38.6       | 67.3      |
| [SAM2.1_small](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_s.pt) | 1024          | 46.1       | 77.4      |
| [SAM2.1_base](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_b.pt)  | 1024          | 80.8       | 148.9     |
| [SAM2.1_large](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_l.pt) | 1024          | 224.5      | 425.2     |
| [MobileSAM](https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt)  | 1024          | 9.66       | -         |
| [FastSAM-s](https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt)   | 1024          | 11.8       | 28.6      |
| [FastSAM-x](https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-x.pt)   | 1024          | 68.2       | 167.2     |


