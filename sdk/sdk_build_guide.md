# SDK Build Guide

## x86

### 依赖库

#### MNN

```bash
# 编译
cd 3rd_party/MNN
mkdir build && cd build
cmake .. \
  -DMNN_SEP_BUILD=OFF \
  -DMNN_BUILD_OPENCV=ON \
  -DMNN_IMGCODECS=ON \
  -DMNN_ARM82=OFF \
  -DMNN_KLEIDIAI=OFF \
  -DCMAKE_BUILD_TYPE=Release
make -j8 MNN

# 复制动态库到依赖库目录
cp libMNN.so ../../../sdk/lib/x86
```

#### yaml-cpp & jsoncpp

```bash
sudo apt install libyaml-cpp-dev libjsoncpp-dev
```

### 编译

```bash
cd sdk
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
```

### 运行Demo

cmake添加`-DBUILD_DEMO=ON`选项重新构建编译，得到可执行文件`DETECT`和`SEGMENT`，生成目录在`sdk/bin/x86`下

```bash
export LD_LIBRARY_PATH=./sdk/bin/x86:$LD_LIBRARY_PATH
./sdk/bin/x86/DETECT models/yolo26x_best.mnn data/test.jpg data/config.yaml
./sdk/bin/x86/SEGMENT models/sam2.1_b_enc.mnn models/sam2.1_b_dec.mnn data/test.jpg 500,720,560,1070 1,1
```

## Android

### NDK

1. 下载[NDK](https://developer.android.google.cn/ndk/downloads?hl=zh-cn)，解压到`/path/to/android-ndk`

2. 在`.bashrc`或者`.bash_profile`中设置NDK环境变量，例如：`export ANDROID_NDK=/path/to/android-ndk`

3. 复制C++动态库到依赖库目录
    ```bash
    cp ${/path/to/android-ndk}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so sdk/lib/android/
    ```

### 依赖库

#### MNN

```bash
cd 3rd_party/MNN
mkdir build_android && cd build_android
cmake .. \
  -DMNN_SEP_BUILD=OFF \
  -DMNN_BUILD_OPENCV=ON \
  -DMNN_IMGCODECS=ON \
  -DMNN_OPENCL=ON \
  -DMNN_VULKAN=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_BUILD_FOR_ANDROID_COMMAND=ON \
  -DMNN_SUPPORT_BF16=ON \
  -DMNN_KLEIDIAI=OFF \
  -DMNN_USE_SSE=OFF \
  -DMNN_USE_LOGCAT=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_STL=c++_shared \
  -DANDROID_NATIVE_API_LEVEL=android-21
make -j8 MNN

# 复制动态库到依赖库目录
cp libMNN.so ../../../sdk/lib/android
```

#### yaml-cpp

```bash
cd 3rd_party/yaml-cpp
mkdir build_android && cd build_android
cmake .. \
  -DYAML_BUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_STL=c++_shared \
  -DANDROID_NATIVE_API_LEVEL=android-21
make -j8

# 复制动态库到依赖库目录
cp libyaml-cpp.so ../../../sdk/lib/android
```

#### jsoncpp

```bash
cd 3rd_party/jsoncpp
mkdir build_android && cd build_android
cmake .. \
  -DJSONCPP_WITH_TESTS=OFF \
  -DBUILD_STATIC_LIBS=OFF \
  -DBUILD_OBJECT_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_STL=c++_shared \
  -DANDROID_NATIVE_API_LEVEL=android-21
make -j8

# 复制动态库到依赖库目录
cp lib/libjsoncpp.so ../../../sdk/lib/android
```

### 编译

```bash
cd sdk
mkdir build_android && cd build_android
cmake .. \
  -DBUILD_FOR_ANDROID=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_STL=c++_shared \
  -DANDROID_NATIVE_API_LEVEL=android-21
make -j8
```

### 运行Demo

1. cmake添加`-DBUILD_DEMO=ON`选项重新构建编译，得到可执行文件`DETECT`和`SEGMENT`，生成目录在`sdk/bin/android`下

2. 下载[官方工具Platform-Tools](https://googledownloads.cn/android/repository/platform-tools-latest-windows.zip)并解压，ADB路径为`${platform-tools}\adb.exe`

3. 推送文件到手机端: 手机连接PC，选择"USB文件传输模式"，设置->开发者选项，打开"USB调试"

    ```powershell
    # 列出已连接设备
    D:\WorkSpace\platform-tools\adb.exe devices
    # List of devices attached
    # ef85cee5        device

    # 推送Demo文件到设备/data/local/tmp
    Get-ChildItem "sdk\bin\android\*" | ForEach-Object { D:\WorkSpace\platform-tools\adb.exe push $_.FullName /data/local/tmp }

    # 推送模型、配置文件和测试图片到设备/data/local/tmp
    Get-ChildItem "models\*.mnn" | ForEach-Object { D:\WorkSpace\platform-tools\adb.exe push $_.FullName /data/local/tmp }
    D:\WorkSpace\platform-tools\adb.exe push data\config.yaml /data/local/tmp
    D:\WorkSpace\platform-tools\adb.exe push data\test.jpg /data/local/tmp
    ```

4. 运行Demo

    ```bash
    # 进入设备
    PS D:\WorkSpace\platform-tools\adb.exe shell

    # 设置执行权限
    cd /data/local/tmp
    chmod +x DETECT SEGMENT

    # 设置动态库目录
    export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

    # 运行
    ./DETECT yolo26x_best.mnn test.jpg config.yaml
    ./SEGMENT sam2.1_b_enc.mnn sam2.1_b_dec.mnn test.jpg 500,720,560,1070 1,1

    # 拉取结果
    PS D:\WorkSpace\platform-tools\adb.exe pull /data/local/tmp/test_det.jpg .\data\
    PS D:\WorkSpace\platform-tools\adb.exe pull /data/local/tmp/test_seg.jpg .\data\
    ```
