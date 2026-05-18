# Railway_Tools_Detection

## 训练

### 环境配置

- conda
- python~=3.12
- torch==2.11.0  # --index-url https://download.pytorch.org/whl/cpu
- torchvision==0.26.0
- ultralytics~=8.4
- onnx
- onnxruntime
- onnxslim
- mnn~=3.5
- aliyun-log-python-sdk
- beautifulsoup4
- pypinyin

### 运行

```bash
# 爬取图像
python 01_crawl.py 老虎钳 --num 50 --output data/downloads

# 模型训练
python 02_train.py --train --config data/config.yaml --model models/yolo26x.pt --model_type yolo26x.yaml --device 0

# 模型导出
python 02_train.py --export_mnn --model models/yolo26x.pt

# pt模型推理
python 03_inference.py --input data/test --model models/yolo26x.pt

# mnn模型推理
python 04_mnn_inference.py --input data/test --config data/config.yaml --model models/yolo26x.mnn

```

## MNN推理

### x86环境配置

#### MNN编译

1. 下载[MNN](https://github.com/alibaba/MNN)源码, 解压到`project_root`
   
2. 编译动态库

```bash
cd /project_root/MNN
mkdir build && cd build && cmake -DCMAKE_CXX_STANDARD=17 -DMNN_IMGCODECS=ON -DMNN_BUILD_OPENCV=ON .. && make -j8
```

3. 复制MNN动态库到项目目录

```bash
find . -name "*.so" -exec cp -t ../../interface/lib {} +
```

#### 项目编译

1. 安装`yaml-cpp`和`jsoncpp

```bash
sudo apt install libjsoncpp-dev libyaml-cpp-dev
```

2. (首次)复制MNN头文件到项目目录

```bash
cp -r MNN/include/* interface/include
cp -r MNN/tools/cv/include/* interface/include/MNN
```

3. 编译项目

```bash
cd /project_root/interface
mkdir build && cd build && cmake .. && make -j8
```