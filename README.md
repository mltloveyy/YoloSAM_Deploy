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

cmake添加`-DBUILD_DEMO=ON`选项重新构建编译，得到可执行文件`MNN_YOLO`，生成目录在`interface/bin/x86`下

```bash
./MNN_YOLO models/yolo26x.mnn data/test.jpg data/config.yaml
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

1. cmake添加`-DBUILD_DEMO=ON`选项重新构建编译，得到可执行文件`MNN_YOLO`，生成目录在`interface/bin/android`下

2. 在Win系统上下载并安装[雷电模拟器](https://www.ldmnq.com/)

3. 打开雷电模拟器，菜单->软件设置->其他，打开ROOT权限，ADB调试选择”开启本地连接“，保存并重启模拟器

4. ADB路径：`${模拟器安装路径}\adb.exe`，默认路径：`C:\leidian\LDPlayer9\adb.exe`

5. 推送文件到模拟器
   
```powershell
# 列出已连接设备
C:\leidian\LDPlayer14\adb.exe devices
# List of devices attached
# emulator-5554   device

# 推送Demo文件到模拟器/data/local/tmp/目录下
Get-ChildItem "interface\bin\android\*" | ForEach-Object { C:\leidian\LDPlayer14\adb.exe push $_.FullName /data/local/tmp/ }

# 推送c++动态库到模拟器/data/local/tmp/目录下
C:\leidian\LDPlayer14\adb.exe push ${/path/to/android-ndk}\toolchains\llvm\prebuilt\linux-x86_64\sysroot\usr\lib\aarch64-linux-android\libc++_shared.so /data/local/tmp/

# 推送模型、配置文件和测试图片到模拟器/data/local/tmp/目录下
C:\leidian\LDPlayer14\adb.exe push runs\detect\20260513\yolo26x_best.mnn /data/local/tmp/
C:\leidian\LDPlayer14\adb.exe push data\config.yaml /data/local/tmp/
C:\leidian\LDPlayer14\adb.exe push data\test.jpg /data/local/tmp/
```

6. 运行Demo
   
```bash
# 进入模拟器
PS C:\leidian\LDPlayer14\adb.exe shell

# 设置执行权限
cd /data/local/tmp
chmod +x MNN_YOLO

# 设置动态库目录
export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH

# 运行
./MNN_YOLO yolo26.mnn test.jpg config.yaml

# 拉取结果
PS C:\leidian\LDPlayer14\adb.exe pull /data/local/tmp/result_test.jpg .\data\
```
