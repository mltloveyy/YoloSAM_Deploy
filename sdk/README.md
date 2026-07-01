# YoloSAM_Deploy SDK

> 本库基于MNN推理引擎实现目标检测和分割，提供C++接口，并通过JNI可在Android中被调用。

## 编译与运行Demo

[Build指南](sdk_build_guide.md)

## 集成

- 复制动态库`libalgorithm.so,libMNN.so,libMNN_Express.so,libMNNOpenCV.so`到`app/src/main/jniLibs/arm64-v8a/`

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
package com.example.segmentor;

public class Detector {
    // 检测结果
    public static class Detection {
        public int x0, y0, x1, y1;
        public int classId;
        public String className;
        public float confidence;
    }

    // 加载模型并初始化
    public native boolean init(String modelPath, int[] classIds, String[] classNames, int forwardType, int precisionMode, int numThreads);

    // 检测图片
    public native Detection[] detect(String imagePath);

    // 释放资源
    public native void release();
}

public class Segmentor {
    // 分割结果
    public static class Segmentation {
        public int x0, y0, x1, y1;
        public int[] points;
    }

    // 加载模型并初始化
    public native boolean init(String encModelPath, String decModelPath, int forwardType, int precisionMode, int numThreads);

    // 加载图片
    public native void load(String imagePath);

    // 根据提示信息分割图片
    public native Segmentation segment(int[] pointCoords, int[] pointLabels);

    // 释放资源
    public native void release();
}
```

### Native 方法与 C++ 接口映射

#### Detector

| Java 方法     | C++ 对应操作            | 说明                     |
| ----------- | ------------------- | ---------------------- |
| init(...)   | 创建 Detector 实例      | 加载模型、类别映射文件、设置推理后端和线程数 |
| detect(...) | 调用 .run(image_path) | 返回检测结果数组               |
| release()   | 销毁 Detector 实例      | 释放内存                   |

#### Segmentor

| Java 方法      | C++ 对应操作                                | 说明                     |
| ------------ | --------------------------------------- | ---------------------- |
| init(...)    | 创建 Segmentor 实例                         | 加载模型、类别映射文件、设置推理后端和线程数 |
| load(...)    | 调用 .set_image(image_path)               | 加载并编码图像                |
| segment(...) | 调用 .forward(point_coords, point_labels) | 输入提示信息，返回分割结果          |
| release()    | 销毁 Segmentor 实例                         | 释放内存                   |

## 接口参数详解

### Detector

#### 1. `init`

```java
boolean init(String modelPath, int[] classIds, String[] classNames, int forwardType, int precisionMode, int numThreads)
```

| 参数            | 类型       | 说明                                                                             |
| ------------- | -------- | ------------------------------------------------------------------------------ |
| modelPath     | String   | mnn检测模型文件路径                                                                    |
| classIds      | int[]    | 类别ID数组，需与模型的类别ID一致，如{0, 1, 2}，参考`data/config.yaml` |
| classNames    | String[] | 类别名称数组，与classIds顺序对应，如{"xiezuiqian", "jianzuiqian", "laohuqian"}               |
| forwardType   | int      | 推理后端类型                                                                         |
| precisionMode | int      | 精度模式                                                                           |
| numThreads    | int      | CPU线程数                                                                         |
| **return**    | boolean  | true表示初始化成功，false失败(如模型文件不存在、配置错误)                                             |

#### 2. `detect`

```java
Detection[] detect(String imagePath)
```

| 参数         | 类型          | 说明                                                                                                                                     |
| ---------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| imagePath  | String      | 图片绝对路径，支持jpg、jpeg、png                                                                                                                  |
| **return** | Detection[] | 检测结果数组，每个元素包含: <br>- `x0, y0, x1, y1`: 边界框左上角、右下角坐标<br>- `classId`: 类别ID<br>- `className`: 类别名称<br>- `confidence`: 置信度<br>若无检测目标，返回空数组 |

#### 3. `release`

释放Native端`Detector`对象，避免内存泄漏。调用后不可再使用`detect`。 

### Segmentor

#### 1. `init`

```java
boolean init(String encModelPath, String decModelPath, int forwardType, int precisionMode, int numThreads)
```

| 参数            | 类型      | 说明                                 |
| ------------- | ------- | ---------------------------------- |
| encModelPath  | String  | mnn编码模型文件路径                        |
| decModelPath  | String  | mnn解码模型文件路径                        |
| forwardType   | int     | 推理后端类型                             |
| precisionMode | int     | 精度模式                               |
| numThreads    | int     | CPU线程数                             |
| **return**    | boolean | true表示初始化成功，false失败(如模型文件不存在、配置错误) |

#### 2. `load`

```java
void load(String imagePath)
```

| 参数        | 类型     | 说明                    |
| --------- | ------ | --------------------- |
| imagePath | String | 图片绝对路径，支持jpg、jpeg、png |

#### 3. `segment`

```java
Segmentation segment(int[] pointCoords, int[] pointLabels);
```

| 参数          | 类型           | 说明                                                                       |
| ----------- | ------------ | ------------------------------------------------------------------------ |
| pointCoords | int[]        | 点坐标数组，如{x1, y1, x2, y2}                                                  |
| pointLabels | int[]        | 点标签数组，长度为pointCoords的1/2，如{1, 1}。正样本点=1，负样本点=0，边界框左上角=2，边界框右下角=3         |
| **return**  | Segmentation | 分割结果，包含: <br>- `x0, y0, x1, y1`: 边界框左上角、右下角坐标<br>- `points`: 轮廓点坐标数组<br> |

#### 4. `release`

释放Native端`Segmentor`对象，避免内存泄漏。调用后不可再使用`load`和`segment`。 

---

## 注意事项

1. **线程安全**: `Detector`和`Segmentor`实例非线程安全，每个线程应创建独立实例，或外部加锁。
2. **性能**: 在推理前对模型进行预热(构造函数中设置 `warmup=true`)可以使推理耗时更加稳定。
