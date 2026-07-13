# Benchmark

[MNN模型压缩指南](https://mnn-docs.readthedocs.io/en/latest/tools/compress.html#id3)

[MNN后端配置](https://mnn-docs.readthedocs.io/en/latest/start/python.html#id15)

## Benckmark工具

```bash
sh benchmark.sh

# 绑定指定CPU核运行
taskset C0 sh benchmark.sh  # cpu6-7
taskset 3F sh benchmark.sh  # cpu0-5
taskset FF sh benchmark.sh  # all
```

## 支持清单

|             | CPU    | GPU                  | NPU   |
| ----------- | ------ | -------------------- | ----- |
| 可用模型        | 全部模型   | 几乎全部模型               | CV模型  |
| 可变形状/控制流    | 支持     | 支持但有性能损失             | 不支持   |
| 加载时间        | 短      | 中 / 长                | 中     |
| 算力          | 中      | 中                    | 高     |
| 功耗          | 高      | 中                    | 低     |
| Forwardtype | CPU(0) | OPENCL(3), VULKAN(7) | NN(5) |

- **PC CPU**(i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0)
  
  - ForwardType: CPU(0)
  
  - PrecisionMode: NORMAL(0)
  
  - MemoryMode: NORMAL(0)

- **DEV**(i8sdot:1, fp16:1, i8mm: 1, sve2: 1, sme2: 0)
  
  - ForwardType: CPU(0), OPENCL(3), VULKAN(7)
  
  - PrecisionMode: NORMAL(0), LOW(2), LOW_BF(3)
  
  - MemoryMode: NORMAL(0), LOW(2)

## 对比结果

### PC

#### FP32权重 + NORMAL精度 + CPU后端 + 不同线程数

- 增加线程数会加速推理，且不会增加内存消耗

|          | Ultralytics(PY) | MNN(PY) 4x | MNN(PY) 2x | MNN(PY) 1x | MNN(CPP) 4x | MNN(CPP) 2x | MNN(CPP) 1x |
| -------- | --------------- | ---------- | ---------- | ---------- | ----------- | ----------- | ----------- |
| YOLO26x  | 0.8s            | 0.9s       | 1.4s       | 2.3s       | 0.9s        | 1.2s        | 1.9s        |
| SAM2.1_b | 35.9s           | 7.6s       | 10.6s      | 12.8s      | 5.6s        | 6.7s        | 9.7s        |
| SAM2.1_s | 32.7s           | 5.0s       | 5.0s       | 6.6s       | 4.0s        | 3.7s        | 5.5s        |
| SAM2.1_t | 32.4s           | 3.7s       | 4.3s       | 6.5s       | 2.8s        | 3.2s        | 4.8s        |

#### 不同量化权重 + NORMAL精度 + CPU后端 + 2x线程数

- 权重量化对模型文件大小压缩明显，对推理精度和速度影响很小，对推理RAM占用的影响较小

| FileSize / RAM (MB) | FP32            | FP16             | INT8             |
| ------------------- | --------------- | ---------------- | ---------------- |
| YOLO26x             | 212 / 870       | 106 / 782        | 53.8 / 730       |
| SAM2.1_b            | 291+23.5 / 2016 | 146+11.8 / 1868  | 95.3+12.1 / 1815 |
| SAM2.1_s            | 155+23.5 / 1202 | 77.8+11.8 / 1084 | 57.6+12.1 / 1070 |
| SAM2.1_t            | 128+23.5 / 1108 | 64.1+11.8 / 1030 | 50.6+12.1 / 1018 |

### DEV

#### CPU后端

##### RAM占用（不同量化权重 + 不同精度 + 2x线程数）

| RAM(MB)       | YOLO26x | SAM2.1_b | SAM2.1_s | SAM2.1_t |
| ------------- | ------- | -------- | -------- | -------- |
| FP32 + Normal | 1143    | 2377     | 1352     | 1250     |
| FP32 + Low    | 742     | 1649     | 943      | 858      |
| FP16 + Normal | 911     | 2023     | 1164     | 1092     |
| FP16 + Low    | 522     | 1287     | 768      | 712      |
| INT8 + Normal | 798     | 1914     | 1120     | 1063     |
| INT8 + Low    | 405     | 1178     | 724      | 683      |

##### 推理速度（不同量化权重 + 不同精度 + 不同线程数）

- CPU动态调频，推理速度仅供参考

|                    | YOLO26x | SAM2.1_b Enc | SAM2.1_s Enc | SAM2.1_t Enc |
| ------------------ | ------- | ------------ | ------------ | ------------ |
| FP32 + Normal + 4x | 0.8s    | 5.4s         | 3.3s         | 2.9s         |
| FP32 + Normal + 2x | 1.1s    | 6.3s         | 4.0s         | 3.4s         |
| FP32 + Normal + 1x | 1.8s    | 10.9s        | 5.3s         | 4.6s         |
| FP32 + Low    + 4x | 0.5s    | 3.3s         | 1.8s         | 1.6s         |
| FP32 + Low    + 2x | 0.6s    | 3.0s         | 1.9s         | 1.8s         |
| FP32 + Low    + 1x | 1.1s    | 9.6s         | 5.4s         | 4.5s         |
| FP16 + Normal + 4x | 0.8s    | 6.9s         | 3.9s         | 3.4s         |
| FP16 + Normal + 2x | 1.0s    | 7.1s         | 4.0s         | 3.6s         |
| FP16 + Normal + 1x | 1.8s    | 10.2s        | 5.5s         | 4.6s         |
| FP16 + Low    + 4x | 0.5s    | 3.2s         | 2.1s         | 1.7s         |
| FP16 + Low    + 2x | 0.6s    | 3.9s         | 2.2s         | 1.8s         |
| FP16 + Low    + 1x | 1.1s    | 5.7s         | 3.1s         | 2.6s         |
| INT8 + Normal + 4x | 0.8s    | 5.9s         | 3.9s         | 3.2s         |
| INT8 + Normal + 2x | 1.1s    | 7.2s         | 4.0s         | 3.4s         |
| INT8 + Normal + 1x | 1.8s    | 9.6s         | 5.5s         | 4.6s         |
| INT8 + Low    + 4x | 0.5s    | 3.0s         | 1.7s         | 1.5s         |
| INT8 + Low    + 2x | 0.6s    | 3.0s         | 1.7s         | 1.8s         |
| INT8 + Low    + 1x | 1.2s    | 5.5s         | 3.1s         | 2.6s         |

#### GPU后端

##### 推理速度（FP16量化权重 + Low精度 + 不同后端 + 不同功能）

- GPU后端的numThread是个mask表示多种功能叠加，参考[GPU numThread设置](https://mnn-docs.readthedocs.io/en/latest/start/python.html#numthread)
  - 1: 默认配置
  - 4: 自动搜索最优内核配置，初始化耗时增加但推理更快
  - 64: buffer 数据类型
  - 128: image 数据类型
  - 512: 批量记录 GPU 命令，高通芯片上可显著提升性能

**功能支持矩阵**

|               | OpenCL | Vulkan |
| ------------- | ------ | ------ |
| 1             | ✅      | ✅      |
| 4             | ✅      | ✅      |
| 64            | ✅      | ❌      |
| 128           | ✅      | ❌      |
| 512           | ❌      | ❌      |
| 64 + 4        | ✅      | ❌      |
| 128 + 4       | ✅      | ❌      |
| 512 + 4       | ❌      | ✅      |
| 512 + 64      | ✅      | ❌      |
| 512 + 128     | ❌      | ❌      |
| 512 + 64 + 4  | ✅      | ❌      |
| 512 + 128 + 4 | ❌      | ❌      |

**YOLO26x**

- GPU动态调频，推理速度仅供参考

| 1st(Avg)     | w/o cache   | with cache |
| ------------ | ----------- | ---------- |
| CPU    + 2x  | 11.5s(0.8s) | -          |
| OpenCL + 1   | 17.7s(1.7s) | 4.3s(1.7s) |
| OpenCL + 4   | >1min(1.6s) | 5.1s(1.6s) |
| OpenCL + 64  | >2min(0.7s) | 1.5s(0.6s) |
| OpenCL + 128 | >1min(1.7s) | 5.4s(1.7s) |
| OpenCL + 68  | >2min(0.7s) | 1.5s(0.6s) |
| OpenCL + 132 | >1min(1.7s) | 5.1s(1.6s) |
| OpenCL + 576 | >2min(0.8s) | 1.7s(0.7s) |
| OpenCL + 580 | >2min(0.8s) | 1.7s(0.7s) |
| Vulkan + 1   | 7.3s(1.9s)  | 7.3s(1.9s) |
| Vulkan + 4   | 22.7s(1.8s) | 9.6s(1.8s) |
| Vulkan + 516 | 22.7s(1.8s) | 9.5s(1.8s) |

#### 测试

- Android设备需要Root情况下才可以手动锁频

##### CPU锁频

```bash
# 查看CPU开启状态
cat /sys/devices/system/cpu/cpu*/online

# 开启所有CPU
for i in 0 1 2 3 4 5 6 7; do echo 1 > /sys/devices/system/cpu/cpu${i}/online 2>/dev/null; done

# 查看CPU0可用频率
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
# 384000 537600 691200 787200 883200 998400 1113600 1228800 1324800 1440000 1555200 1670400 1785600 1900800 1996800
# 2112000 2227200 2361600 2496000 2611200 2745600 2899200 3033600 3187200 3302400 3398400 3513600 3628800

# 查看CPU0当前最小/最大频率
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq

# 修改CPU0最小/最大频率
echo 3628800 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
echo 3628800 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq

# 查看CPU0当前频率"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

##### GPU锁频

```bash
# 查看GPU可用频率
cat /sys/class/kgsl/kgsl-3d0/gpu_available_frequencies

# 修改GPU最小/最大频率
echo 900000000 > /sys/class/kgsl/kgsl-3d0/min_clock_mhz
echo 900000000 > /sys/class/kgsl/kgsl-3d0/max_clock_mhz
```