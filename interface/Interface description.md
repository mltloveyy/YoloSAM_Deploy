# 检测接口说明文档

## 概述

本库基于 MNN 框架实现 YOLO 目标检测，提供 C++ 接口，并支持通过 JNI 在 Android 平台（Java/Kotlin）中调用。  
核心类为 `Detector`，负责模型加载、推理和后处理。

## 集成

- 复制动态库`libdetector.so,libMNN.so,libMNN_Express.so,libMNNOpenCV.so`到 `app/src/main/jniLibs/arm64-v8a/`

- 在 Java 代码中加载动态库：
  
  ```java
  static {
      System.loadLibrary("detector");
  }
  ```

## JNI 接口说明

以下为推荐的 Java 层封装类及其 Native 方法声明。实际 JNI 实现需在 C++ 侧完成，映射到 `Detector` 类的操作。

### Java 类定义

```java
package com.example.detector;

public class Detector {
    // 用于保存检测结果
    public static class Detection {
        public float x0, y0, x1, y1;   // 边界框坐标（像素）
        public int classId;
        public String className;
        public float confidence;
    }

    // 加载模型并初始化
    public native boolean init(String modelPath, int[] classIds, String[] classNames,
                               int forwardType, int precisionMode, int numThreads);

    // 对单张图片进行检测（图片路径）
    public native Detection[] detectImage(String imagePath);

    // 对单张图片进行检测（传入像素数据，需自行适配）
    // public native Detection[] detectBitmap(Bitmap bitmap); // 可选

    // 释放资源
    public native void release();

    // 单例或实例管理请自行封装
}
```

### Native 方法与 C++ 接口映射

| Java 方法          | C++ 对应操作                    | 说明                     |
| ---------------- | --------------------------- | ---------------------- |
| init(...)        | 创建 Detector实例               | 加载模型、类别映射文件、设置推理后端和线程数 |
| detectImage(...) | 调用 detector.run(image_path) | 返回检测结果数组               |
| release()        | 销毁 Detector实例               | 释放内存                   |

## 接口参数详解

### 1. `init`

```java
boolean init(String modelPath, String configPath, int forwardType, int precisionMode, int numThreads)
```

| 参数            | 类型       | 说明                                                                  |
| ------------- | -------- | ------------------------------------------------------------------- |
| modelPath     | String   | mnn模型文件的绝对路径                                                        |
| classIds      | int[]    | 类别 ID 数组，如 {0, 1, 2}。需与模型训练时的 ID 一致                                 |
| classNames    | String[] | 类别名称数组，与 classIds 顺序对应，如 {"xiezuiqian", "jianzuiqian", "laohuqian"} |
| forwardType   | int      | 推理后端类型                                                              |
| precisionMode | int      | 精度模式                                                                |
| numThreads    | int      | CPU 线程数（仅 CPU 后端生效）                                                 |
| **返回值**       | boolean  | true表示初始化成功，false失败（如模型文件不存在、配置错误）                                  |

### 2. `detectImage`

```java
Detection[] detectImage(String imagePath)
```

| 参数        | 类型          | 说明                                                                                                                                      |
| --------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| imagePath | String      | 图片绝对路径，支持jpg、jpeg、png                                                                                                                   |
| **返回值**   | Detection[] | 检测结果数组，每个元素包含：<br>- `x0, y0, x1, y1`：边界框左上角、右下角坐标（像素）<br>- `classId`：类别索引（对应配置文件中的 ID）<br>- `className`：类别名称<br>- `confidence`：置信度（0~1） |

若无检测目标，返回空数组（不是 null）。

### 3. `release`

释放 Native 侧 `Detector` 对象，避免内存泄漏。调用后不可再使用 `detectImage`。 

---

## 注意事项

1. **线程安全**：`Detector` 实例非线程安全，每个线程应创建独立实例，或外部加锁。
2. **性能**：首次推理会进行模型加载和预热（构造函数中 `warmup=true`），之后 `detectImage` 耗时稳定。

---

## 附录：配置文件格式 (config.yaml)

```yaml
names:
  0: xiezuiqian
  1: jianzuiqian
  2: laohuqian
  # ... 按模型实际类别填写
```


