# Railway_Tools_Detection

## Python训练与推理

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
python 03_inference.py --model models/yolo26x.pt --input data/test

# mnn模型推理
python 04_mnn_inference.py --model models/yolo26x.mnn --input data/test --config data/config.yaml

```

## C++推理

### x86

#### 依赖库

##### MNN

```bash
# 编译
cd 3rd_party/MNN
mkdir build && cd build
cmake .. -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DCMAKE_BUILD_TYPE=Release && make -j8

# 复制动态库到依赖库目录
find . -name "*.so" -exec cp -t ../../../interface/lib/x86 {} +
```

##### yaml-cpp & jsoncpp

```bash
sudo apt install libyaml-cpp-dev libjsoncpp-dev
```

#### 编译

```bash
cd interface
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
```

#### 运行Demo

cmake添加`-DBUILD_DEMO=ON`选项重新构建编译

```bash
./MNN_YOLO models/yolo26x.mnn data/test data/config.yaml
```

### Android

#### 依赖库

##### MNN

1. 下载[NDK](https://developer.android.google.cn/ndk/downloads?hl=zh-cn), 解压到`/path/to/android-ndk`

2. 在`.bashrc`或者`.bash_profile`中设置NDK环境变量，例如：`export ANDROID_NDK=/path/to/android-ndk`
   
3. 编译

```bash
cd 3rd_party/MNN
mkdir build_android && cd build_android
cmake .. -DMNN_BUILD_OPENCV=ON \
  -DMNN_IMGCODECS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_STL=c++_shared \
  -DANDROID_NATIVE_API_LEVEL=android-21 \
  -DMNN_BUILD_FOR_ANDROID_COMMAND=ON \
  -DMNN_USE_SSE=OFF \
  -DMNN_USE_LOGCAT=OFF
make -j8
```

4. 复制动态库到依赖库目录

```bash
find . -name "*.so" -exec cp -t ../../../interface/lib/android {} +
```

##### yaml-cpp

```bash
cd 3rd_party/yaml-cpp
mkdir build_android && cd build_android
cmake .. -DYAML_BUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_STL=c++_shared \
  -DANDROID_NATIVE_API_LEVEL=android-21
make -j8

# 复制动态库到依赖库目录
cp libyaml-cpp.so ../../../interface/lib/android
```

##### jsoncpp

```bash
cd 3rd_party/jsoncpp
mkdir build_android && cd build_android
cmake .. -DJSONCPP_WITH_TESTS=OFF \
  -DBUILD_STATIC_LIBS=OFF \
  -DBUILD_OBJECT_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_STL=c++_shared \
  -DANDROID_NATIVE_API_LEVEL=android-21
make -j8

# 复制动态库到依赖库目录
cp lib/libjsoncpp.so ../../../interface/lib/android
```

#### 编译

```bash
cd interface
mkdir build_android && cd build_android
cmake .. -DBUILD_FOR_ANDROID=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_STL=c++_shared \
  -DANDROID_NATIVE_API_LEVEL=android-21
make -j8
```

#### 运行Demo

1. cmake添加-DBUILD_DEMO=ON选项重新构建编译
