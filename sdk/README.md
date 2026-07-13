# YoloSAM_Deploy SDK

> 本库基于MNN推理引擎实现目标检测和分割，提供C++接口，并通过JNI可在Android中被调用。

## 编译与运行Demo

[Build指南](sdk_build_guide.md)

## 集成

- 复制库文件到 `app/src/main/jniLibs/arm64-v8a/`
  
  - libalgorithm.so
  - libMNN.so
  - libc++_shared.so

- 在Java代码中加载动态库:
  
  ```java
  static {
      System.loadLibrary("algorithm");
  }
  ```

## JNI 接口说明

以下为推荐的Java层封装类及其Native方法声明。

### Java 类定义示例

```java
package com.example.detector;

public class Detector {
    // 配置
    public static class DetectorConfig {
        public String modelPath;
        public int[] classIds;
        public String[] classNames;
        public int forwardType = 0;
        public int numThreads = 1;
        public int precisionMode = 0;
        public int memoryMode = 0
        public boolean warmup = false;
    }

    // 检测结果
    public static class DetectionResult {
        public int x0, y0, x1, y1;
        public int classId;
        public String className;
        public float confidence;
    }

    // 加载模型并初始化
    public native boolean init(DetectorConfig config);

    // 检测图片
    public native DetectionResult[] detect(String imagePath);

    // 释放资源
    public native void release();
}
```

```java
package com.example.segmentor;

public class Segmentor {
    // 配置
    public static class SegmentorConfig {
        public String modelPath;
        public int forwardType = 0;
        public int numThreads = 1;
        public int precisionMode = 0;
        public int memoryMode = 0;
        public boolean warmup = false;
    }

    // 分割结果
    public static class SegmentationResult {
        public int x0, y0, x1, y1;
        public int[] points;
    }

    // 加载模型并初始化
    public native boolean init(SegmentorConfig encConfig, SegmentorConfig decConfig);

    // 加载图片
    public native void load(String imagePath);

    // 根据提示信息分割图片
    public native SegmentationResult segment(int[] pointCoords, int[] pointLabels);

    // 释放资源
    public native void release();
}
```

### Native 方法与 C++ 接口映射

#### Detector

| Java 方法      | C++ 对应操作                         | 说明                                                         |
| ------------ | -------------------------------- | ---------------------------------------------------------- |
| init(config) | 创建 `Detector(DetectorConfig)` 实例 | 将Java `DetectorConfig` 转为C++ `DetectorConfig` 结构体，加载模型并初始化 |
| detect(...)  | 调用 `.run(image_path)`            | 返回 `DetectionResult` 数组                                    |
| release()    | 销毁 Detector 实例                   | 释放内存                                                       |

#### Segmentor

| Java 方法                    | C++ 对应操作                                            | 说明                                                 |
| -------------------------- | --------------------------------------------------- | -------------------------------------------------- |
| init(encConfig, decConfig) | 创建 `Segmentor(SegmentorConfig, SegmentorConfig)` 实例 | 将Java两个 `SegmentorConfig` 转为C++结构体，加载编码器/解码器模型并初始化 |
| load(...)                  | 调用 `.set_image(image_path)`                         | 加载并编码图像                                            |
| segment(...)               | 调用 `.forward(point_coords, point_labels)`           | 输入提示信息，返回 `SegmentResult`                          |
| release()                  | 销毁 Segmentor 实例                                     | 释放内存                                               |

## 接口参数详解

### Detector

#### 1. `init`

```java
boolean init(DetectorConfig config)
```

**DetectorConfig 字段说明**

| 参数            | 类型       | 默认值   | 说明                                                                      |
| ------------- | -------- | ----- | ----------------------------------------------------------------------- |
| modelPath     | String   | -     | mnn检测模型文件路径                                                             |
| classIds      | int[]    | -     | 类别ID数组，需与模型的类别ID一致，如 `{0, 1, 2}`，参考`data/config.yaml`                   |
| classNames    | String[] | -     | 类别名称数组，与 `classIds` 顺序对应，如 `{"xiezuiqian", "jianzuiqian", "laohuqian"}` |
| forwardType   | int      | 0     | 推理后端类型                                                                  |
| numThreads    | int      | 1     | CPU线程数                                                                  |
| precisionMode | int      | 0     | 精度模式                                                                    |
| memoryMode    | int      | 0     | 内存模式                                                                    |
| warmUp        | boolean  | false | 是否预热模型                                                                  |
| **return**    | boolean  | -     | true表示初始化成功，false失败(如模型文件不存在、配置错误)                                      |

- JNI层将 `classIds` 与 `classNames` 合并为 C++ `std::map<int, std::string> class_names`

#### 2. `detect`

```java
DetectionResult[] detect(String imagePath)
```

| 参数         | 类型                | 说明                                                                                                                                     |
| ---------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| imagePath  | String            | 图片绝对路径，支持jpg、jpeg、png                                                                                                                  |
| **return** | DetectionResult[] | 检测结果数组，每个元素包含: <br>- `x0, y0, x1, y1`: 边界框左上角、右下角坐标<br>- `classId`: 类别ID<br>- `className`: 类别名称<br>- `confidence`: 置信度<br>若无检测目标，返回空数组 |

#### 3. `release`

释放Native端 `Detector` 对象，避免内存泄漏。调用后不可再使用 `detect`。 

### Segmentor

#### 1. `init`

```java
boolean init(SegmentorConfig encConfig, SegmentorConfig decConfig)
```

**SegmentorConfig 字段说明**

| 参数            | 类型      | 默认值   | 说明                                 |
| ------------- | ------- | ----- | ---------------------------------- |
| modelPath     | String  | -     | mnn编码或解码模型文件路径                     |
| forwardType   | int     | 0     | 推理后端类型                             |
| numThreads    | int     | 1     | CPU线程数                             |
| precisionMode | int     | 0     | 精度模式                               |
| memoryMode    | int     | 0     | 内存模式                               |
| warmUp        | boolean | false | 是否预热模型                             |
| **return**    | boolean | -     | true表示初始化成功，false失败(如模型文件不存在、配置错误) |

#### 2. `load`

```java
void load(String imagePath)
```

| 参数        | 类型     | 说明                    |
| --------- | ------ | --------------------- |
| imagePath | String | 图片绝对路径，支持jpg、jpeg、png |

#### 3. `segment`

```java
SegmentationResult segment(int[] pointCoords, int[] pointLabels);
```

| 参数          | 类型                 | 说明                                                                                              |
| ----------- | ------------------ | ----------------------------------------------------------------------------------------------- |
| pointCoords | int[]              | 点坐标数组，如 `{x1, y1, x2, y2}`                                                                      |
| pointLabels | int[]              | 点标签数组，长度为pointCoords的1/2，如 `{1, 1}`。正样本点=1，负样本点=0，边界框左上角=2，边界框右下角=3 |
| **return**  | SegmentationResult | 分割结果，包含: <br>- `x0, y0, x1, y1`: 边界框左上角、右下角坐标<br>- `points`: 轮廓点坐标数组<br>                        |

#### 4. `release`

释放Native端 `Segmentor` 对象，避免内存泄漏。调用后不可再使用 `load` 和 `segment`。 

---

## 注意事项

1. **线程安全**: `Detector` 和 `Segmentor` 实例非线程安全，每个线程应创建独立实例，或外部加锁。
2. **性能**: 在推理前对模型进行预热(构造函数中设置 `warmup=true`)可以使推理耗时更加稳定。
3. **Config 映射**: Java 层的 `DetectorConfig.classIds` + `classNames` 在 JNI 层会被合并为 C++ 的 `std::map<int, std::string> class_names`，两个数组长度必须一致。
4. **Segmentor 双配置**: 编码器和解码器各自拥有独立的 `SegmentorConfig`，可以分别设置不同的推理后端、线程数等参数。
