# Xavier NX YOLO26s 白色半管三维小球位置

当前默认功能已经升级为 RGB-D `tube-v2`：以 25.0 cm 白色开放半管中心为
`0.00 cm`，图像右端为 `+12.50 cm`，输出球沿三维管轴的有符号厘米位置。
厂家内参、畸变和深度到彩色外参均由 Gemini 336L 实时读取；管前后翘起时
不再使用 RGB 像素比例。完整接线、采集、标定及验收流程见
[`TUBE_RGBD.md`](TUBE_RGBD.md)。

另外提供完全不读取深度的纯 RGB 轮廓投影版：
[`TUBE_RGB.md`](TUBE_RGB.md)。它使用独立的 `debug_rgb.py`、`run_rgb.py`，
把实时轮廓两端定义为 `-12.5/+12.5 cm`，将球心投影到二维管轴后线性换算，
适合当前约 14 cm 的近距离安装。

本工程在 Jetson Xavier NX 上训练并部署 YOLO26s，读取奥比中光 Gemini
336L 的 `1280×800@60 MJPEG` 彩色流和 `640×400@30` 深度流，默认输出
半管轴向厘米位置；`--protocol pixel-v1` 可兼容旧的二维球心协议。
原始数据目录 `../data` 只读，所有生成数据、权重、Engine 和报告均保存在本工程。

滚球闭环的 C++17 NX 控制项目位于 [`nx_control/`](nx_control/README.md)。视觉
进程可用 `--control-udp 127.0.0.1:29001` 非阻塞发送带曝光时刻的22字节
`tube-v3`；原有18字节 `tube-v2` 串口输出保持兼容。

## 当前平台与设计

- JetPack 5.1.4、CUDA 11.4、TensorRT 8.5.2、Python 3.8。
- 训练环境固定 Ultralytics 8.4.102 和 NVIDIA CUDA PyTorch 2.1。
- 保留 JetPack 自带、支持 GStreamer 的 OpenCV 4.2，不安装
  `opencv-python`。
- 推理使用 YOLO26 端到端静态 TensorRT FP16 Engine，无 NMS。
- 实测 RGB-D 并行后默认输入 `1×3×416×640`；768×480 未达到 58 Hz 门槛。
- 采集、管姿态、推理、串口相互解耦；每路均为单槽最新值，不累计旧数据。
- CUDA 完成 letterbox、BGR→RGB、归一化和 CHW；固定页锁定缓冲区并尝试
  CUDA Graph。
- Orbbec SDK 获取同步 RGB-D 与厂家标定，彩色 MJPEG 使用 `nvjpegdec`。

TensorRT Engine 绑定本机 TensorRT/CUDA/GPU，不应复制到其他 Jetson 使用。

## 数据集

`python3 prepare_dataset.py` 会完整校验重新标定的数据并生成可复现划分：

| 划分 | 图片 | 正样本 | 负样本 | 球框 |
|---|---:|---:|---:|---:|
| Train | 55 | 50 | 5 | 274 |
| Validation | 17 | 15 | 2 | 105 |
| Test | 17 | 15 | 2 | 104 |

训练集另外生成 55 张确定性的亮度、对比度、阴影和轻度方向性运动模糊图。
验证集和测试集不增强。划分按采集时间连续块完成，详细源路径及图像/标签
SHA-256 在 `dataset/split_manifest.json`。9 张负样本全部来自同一连续片段，因此
负背景泛化必须在真实现场复验。

重新生成数据：

```bash
cd /home/d/2026ti/2026H/ball_yolo26s_nx
python3 prepare_dataset.py
```

## 训练、导出和 Engine 选择

首次建立训练环境：

```bash
bash tools/download_weights.sh
bash tools/setup_train_env.sh
```

正式训练参数已写入 `train.py`：640、150 epochs、batch 2、AMP、workers 2、
patience 30、close mosaic 15、seed 2026。只有发生 CUDA OOM 时才自动用
batch 1 重试。

```bash
./.venv-train/bin/python train.py
./.venv-train/bin/python evaluate_model.py
./.venv-train/bin/python export_models.py
./.venv-train/bin/python evaluate_onnx.py
./.venv-train/bin/python tools/build_engines.py
python3 evaluate_engines.py
python3 select_default.py
```

最后一步只接受以下候选：

- PyTorch、ONNX、TensorRT 的 P/R/mAP50 均不低于 0.95；
- TensorRT Recall 相对同尺寸 PyTorch 下降不超过 0.01；
- `trtexec` 纯推理不低于 75 FPS；
- 若已有相机实测报告，还要求采集中位数不低于 59 FPS、推理平均不低于
  58 FPS、端到端延迟 P95 不高于 35 ms。

通过门槛的最高分辨率写入 `models/default.json`。配置分别记录
`provisional_without_live_camera`、`provisional_without_10min_live` 和
`provisional_without_tube_scene`，短测不能冒充十分钟或白管精度验收。
TensorRT 8.5 ONNX 兼容修改只发生在临时副本，原始 ONNX 会保留。

本机已完成的实际离线结果如下：

| 输入 | TensorRT P/R/mAP50 | 纯推理 FPS | GPU 均值 |
|---|---|---:|---:|
| 768×480 | 1.000 / 0.952 / 0.9993 | 76.247 | RGB-D 实测 56.04 Hz |
| 640×416 | 1.000 / 0.971 / 0.9999 | 92.579 | RGB-D 实测 59.25 Hz |

两个 FP16 Engine 相对 PyTorch 的 Recall 下降均为 0。336L 已验证固件
`1.6.00`、USB 3.2、硬件 JPEG 解码、60/30 RGB-D 和厂家标定。640×416
短测达到 Capture 60.03 Hz、Inference 59.25 Hz、Tube pose 30.01 Hz、
P95 26.29 ms，已写入默认配置；实际白管的十分钟和精度验收仍待完成。

训练输出位于 `artifacts/training/yolo26s_ball_batch*/`，包括 `best.pt`、
`last.pt`、训练参数、曲线、混淆矩阵和批次图。部署副本为
`models/ball_yolo26s_best.pt`。阈值在验证集上选择：优先取 Recall≥0.97 时
Precision 最高点，否则取最大 F1 点。

## Debug 模式

默认不开串口：

```bash
python3 debug.py
```

串口联调：

```bash
python3 debug.py --serial auto --preview-scale 0.75
```

窗口显示白管轮廓、三维轴、`-12.5/0/+12.5` 刻度、球框、球心、轴上投影点
和有符号厘米位置，以及：

- `measured / predicted / lost`；
- 采集帧号、推理帧号、帧龄和累计跳帧；
- Capture/Inference FPS；
- Depth/Tube-pose FPS、姿态倾角、深度年龄、拟合 RMS、有效分箱比例和管置信度；
- 预处理、GPU、后处理、总推理、端到端延迟和滚动 P95；
- Engine 尺寸、CUDA 快速路径、NX 功耗模式、串口设备和发送频率。

按 `q` 或 `Esc` 退出。工程不设置固定 ROI。

纯 RGB、无深度调试：

```bash
python3 debug_rgb.py
```

该入口显示绿色 `RGB CONTOUR SCALE (cm)`，并默认关闭彩色自动曝光，将曝光
设为 SDK 出厂默认值的 30%。实际曝光值和增益显示在窗口顶栏及启动日志。

## Run 模式

建议在正式测速前将 NX 切到允许的最高性能模式并执行 `jetson_clocks`。Run
无窗口，串口默认依次自动寻找 `/dev/ttyACM*`、`/dev/ttyUSB*`：

```bash
python3 run.py
```

无串口运行：

```bash
python3 run.py --no-serial
```

纯 RGB 正式运行：

```bash
python3 run_rgb.py
```

常用参数：

```text
--engine PATH       指定本机 TensorRT Engine
--device auto       仅 V4L2 兼容模式使用；SDK 模式忽略
--conf FLOAT        覆盖验证集选出的阈值
--serial auto|PATH  自动发现或指定串口
--no-serial         禁止串口发送
--baud 921600       8N1 波特率
--stats-interval 2  stderr 统计周期
--preview-scale N   仅 Debug 窗口缩放
```

相机或串口断开后会循环重连。`SIGINT`、`SIGTERM`、`Ctrl+C` 均安全退出。

### 336L 相机节点检查

默认 `--camera-backend sdk` 通过官方 SDK 直接取得 RGB-D，不依赖 RGB
`/dev/videoN` 是否被内核注册。检查完整 SDK 链路：

```bash
python3 tools/check_rgbd_camera.py --frames 180
```

只有使用旧兼容模式 `--protocol pixel-v1 --camera-backend v4l2` 时，才需要
下面的 MJPEG UVC 节点检查：

```bash
python3 tools/check_camera.py
```

如果提示 “MJPEG RGB UVC capture node is not registered”，说明 USB 设备本身
已经连接，但 Linux 内核只注册了深度/红外节点。`/dev/video0` 的
NV12/GRAY8 和 `/dev/video2` 的 Bayer/YV12 不能作为本工程的彩色输入，
不要用 `--device` 强行指定它们。先停止程序，重新插拔直连 NX 的 USB 3
数据口，并检查：

```bash
lsusb -t
gst-device-monitor-1.0 Video/Source
python3 tools/check_camera.py
```

`lsusb -t` 应显示 `5000M`，设备监视器应出现带 `image/jpeg`、
`1280×800@60` 的额外 RGB 节点。仍不出现时重启 NX；若重启后仍缺失，
应使用官方 Orbbec Viewer/SDK 检查固件。Gemini 336L 的 SDK 最低固件为
1.2.20，官方当前推荐 1.6.00。不要把深度或红外节点误当成 RGB。

## 单球跟踪

首次检测选择最高置信度目标。后续根据恒速预测、中心距离、IoU 和框尺寸关联：

- 基础关联距离 80 像素；
- 观测平滑权重 0.75；
- 允许 2 个推理帧使用预测结果；
- 第 3 个未匹配推理帧输出 `lost`；
- 丢失后重新选择最高置信度目标。

参数可用 `--track-distance`、`--track-alpha`、`--hold-frames` 调整。

## USB 串口协议

默认是固定 18 字节小端 `tube-v2`：

| 偏移 | 长度 | 字段 | 说明 |
|---:|---:|---|---|
| 0 | 2 | Magic | `A5 5A` |
| 2 | 1 | Version | `2` |
| 3 | 2 | Payload length | `11` |
| 5 | 4 | Frame ID | `uint32` |
| 9 | 1 | Status | 0 无有效位置、1 实测、2 短时预测 |
| 10 | 2 | Position | `int16`，厘米×100，范围限制为 ±1250 |
| 12 | 2 | Ball confidence | `uint16`，0–1000 |
| 14 | 2 | Tube confidence | `uint16`，0–1000 |
| 16 | 2 | CRC | CRC-16/CCITT-FALSE |

球或管姿态任一无效时，Status、Position 和两个置信度全部置零。CRC 覆盖
字节 2–15。旧像素包用 `--protocol pixel-v1 --camera-backend v4l2`。

旧 `pixel-v1` 字段如下：

| 偏移 | 长度 | 字段 | 说明 |
|---:|---:|---|---|
| 0 | 2 | Magic | `A5 5A` |
| 2 | 1 | Version | `1` |
| 3 | 2 | Payload length | `11` |
| 5 | 4 | Frame ID | `uint32` |
| 9 | 1 | Status | 0 无球、1 实测、2 短时预测 |
| 10 | 2 | X | `uint16`，完整彩色图左上角为原点 |
| 12 | 2 | Y | `uint16` |
| 14 | 2 | Confidence | `uint16`，0–1000 |
| 16 | 2 | CRC | CRC-16/CCITT-FALSE |

CRC 覆盖字节 2–15（Version 至 Payload）。无球包的 X、Y、Confidence 强制为
0。串口阻塞时，新包覆盖未发送旧包，不阻塞推理线程。

## 硬件验收

接入 336L 后分别生成候选 Engine 的 10 分钟实时报告：

```bash
python3 run.py --engine models/ball_yolo26s_768x480_fp16.engine \
  --no-serial --max-seconds 600 \
  --metrics-json artifacts/live_768x480.json

python3 run.py --engine models/ball_yolo26s_640x416_fp16.engine \
  --no-serial --max-seconds 600 \
  --metrics-json artifacts/live_640x416.json

python3 hardware_acceptance.py --mode performance artifacts/live_768x480.json
python3 evaluate_engines.py
python3 select_default.py
```

运动球连续 5 分钟和静止球 1 分钟也用 `--metrics-json` 保存后检查：

```bash
python3 hardware_acceptance.py --mode motion artifacts/motion_5min.json
python3 hardware_acceptance.py --mode static artifacts/static_1min.json
```

报告包含实测检测覆盖率、含短时预测的有效输出率、最大连续预测帧和球心
P95 抖动半径。最大连续预测帧由代码和单元测试保证不超过 2；移走球后第 3
个未匹配推理帧进入无球状态。

## 测试

```bash
python3 -m unittest discover -v
python3 -m compileall -q .
```

测试覆盖数据划分与标签、坐标还原、YOLO26 端到端输出解析、单球跟踪、短时
丢失、CRC、伪串口、最新值队列、相机管线和硬件验收规则。

## 主要报告

- `dataset/split_manifest.json`：可复现划分和源文件哈希；
- `artifacts/training/yolo26s_ball_batch2/`：训练曲线、混淆矩阵和权重；
- `artifacts/pytorch_metrics.json`：PyTorch 阈值及测试指标；
- `artifacts/onnx_metrics.json`：两个 ONNX 的测试指标；
- `artifacts/engine_manifest.json`：本机 Engine 构建信息；
- `artifacts/candidate_report.json`：三后端精度及纯推理性能对比；
- `models/default.json`：通过离线门槛的暂定 Engine 与阈值；
- `VALIDATION.md`：实际离线结果及待接入硬件的验收清单。
