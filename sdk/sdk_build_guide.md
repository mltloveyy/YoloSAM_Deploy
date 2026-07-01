# SDK Build Guide

## x86

### 依赖库

#### MNN

```bash
# 编译
cd 3rd_party/MNN
mkdir build && cd build
cmake .. -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DCMAKE_BUILD_TYPE=Release && make -j8

# 复制动态库到依赖库目录
find . -name "*.so" -exec cp -t ../../../sdk/lib/x86 {} +
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
./sdk/bin/x86/DETECT models/yolo26x.mnn data/test.jpg data/config.yaml
./sdk/bin/x86/SEGMENT models/sam2.1_b_enc.mnn models/sam2.1_b_dec.mnn data/test.jpg 500,720,560,1070 1,1
```

## Android

### NDK

1. 下载[NDK](https://developer.android.google.cn/ndk/downloads?hl=zh-cn), 解压到`/path/to/android-ndk`

2. 在`.bashrc`或者`.bash_profile`中设置NDK环境变量，例如：`export ANDROID_NDK=/path/to/android-ndk`

### 依赖库

#### MNN

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

# 复制动态库到依赖库目录
find . -name "*.so" -exec cp -t ../../../sdk/lib/android {} +
```

#### yaml-cpp

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
cp libyaml-cpp.so ../../../sdk/lib/android
```

#### jsoncpp

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
cp lib/libjsoncpp.so ../../../sdk/lib/android
```

### 编译

```bash
cd sdk
mkdir build_android && cd build_android
cmake .. -DBUILD_FOR_ANDROID=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_STL=c++_shared \
  -DANDROID_NATIVE_API_LEVEL=android-21
make -j8
```

### 运行Demo

1. cmake添加`-DBUILD_DEMO=ON`选项重新构建编译，得到可执行文件`DETECT`和`SEGMENT`，生成目录在`sdk/bin/android`下

2. 下载安装Win版ADB工具(模拟器or官方工具)
   
   - `模拟器`: 下载安装[雷电模拟器](https://www.ldmnq.com/)，ADB路径为`${模拟器安装路径}\adb.exe`，默认路径为`C:\leidian\LDPlayer9\adb.exe`
   
   - `官方工具`: 下载[Platform-Tools](https://googledownloads.cn/android/repository/platform-tools-latest-windows.zip)并解压，ADB路径为`${platform-tools}\adb.exe`

3. 推送文件到模拟器(以模拟器的ADB为例)

    打开雷电模拟器，菜单->软件设置->其他，打开ROOT权限，ADB调试选择"开启本地连接"，保存并重启模拟器

    ```powershell
    # 列出已连接设备
    C:\leidian\LDPlayer14\adb.exe devices
    # List of devices attached
    # emulator-5554   device

    # 推送Demo文件到模拟器/data/local/tmp/目录下
    Get-ChildItem "sdk\bin\android\*" | ForEach-Object { C:\leidian\LDPlayer14\adb.exe push $_.FullName /data/local/tmp/ }

    # 推送libc++_shared.so到模拟器/data/local/tmp/目录下
    C:\leidian\LDPlayer14\adb.exe push ${/path/to/android-ndk}\toolchains\llvm\prebuilt\linux-x86_64\sysroot\usr\lib\aarch64-linux-android\libc++_shared.so /data/local/tmp/

    # 推送模型、配置文件和测试图片到模拟器/data/local/tmp/目录下
    C:\leidian\LDPlayer14\adb.exe push models\yolo26x.mnn /data/local/tmp/
    C:\leidian\LDPlayer14\adb.exe push models\sam2.1_b_enc.mnn /data/local/tmp/
    C:\leidian\LDPlayer14\adb.exe push models\sam2.1_b_dec.mnn /data/local/tmp/
    C:\leidian\LDPlayer14\adb.exe push data\config.yaml /data/local/tmp/
    C:\leidian\LDPlayer14\adb.exe push data\test.jpg /data/local/tmp/
    ```

4. 运行Demo

    ```bash
    # 进入模拟器
    PS C:\leidian\LDPlayer14\adb.exe shell

    # 设置执行权限
    cd /data/local/tmp
    chmod +x DETECT SEGMENT

    # 设置动态库目录
    export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

    # 运行
    ./DETECT yolo26x.mnn test.jpg config.yaml
    ./SEGMENT sam2.1_b_enc.mnn sam2.1_b_dec.mnn test.jpg 500,720,560,1070 1,1

    # 拉取结果
    PS C:\leidian\LDPlayer14\adb.exe pull /data/local/tmp/test_det.jpg .\data\
    PS C:\leidian\LDPlayer14\adb.exe pull /data/local/tmp/test_seg.jpg .\data\
    ```
