# YoloSAM_Deploy

> YoloSAM_Deploy: Cross-platform YOLO & SAM inference with Ultralytics (Python) and MNN (Python/C++) on PC and Mobile

## Python训练与推理

### 环境配置

```bash
pip install -r requirements.txt
```

### 运行

#### 检测

```bash
# 模型训练
python scripts/detect/00_train.py --config data/config.yaml --model models/yolo26x.pt --model_type yolo26x.yaml --device 0

# pt模型推理
python scripts/detect/01_eval.py --model models/yolo26x.pt --input data

# 导出mnn模型(fp16)
python scripts/detect/02_convert.py --model models/yolo26x.pt --quantize 16

# mnn模型推理
python scripts/detect/03_mnn_eval.py --model models/yolo26x.mnn --input data --config data/config.yaml
```

#### 分割

```bash
# pt模型推理
python scripts/segment/01_eval.py --model models/sam2.1_b.pt --image data/test.jpg --points 500 720 560 1070

# 导出mnn模型(fp16)
python scripts/segment/02_convert.py --model models/sam2.1_b.pt --quantize 16

# mnn模型推理
python scripts/segment/03_mnn_eval.py --enc models/sam2.1_b_enc.mnn --dec models/sam2.1_b_dec.mnn --image data/test.jpg --points 500 720 560 1070
```

#### 其他

```bash
# 爬取图像
python scripts/data_process/crawl.py 老虎钳 --num 50 --output data/downloads

# 标注图像
python scripts/annotation.py --model models/sam2.1_b.pt --image data/test.jpg
```

## C++推理

[C++指南](sdk/README.md)
