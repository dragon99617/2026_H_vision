# Gemini 336L 墨绿色/白色半管三维标定与验收

## 坐标定义与运行

开放半管的物理长度固定为 25.0 cm。程序拟合的是半管两侧三维边线的
平均轴线，轴中点为 `0.00 cm`；投影横坐标较小的画面左端固定为正端
`+12.50 cm`，画面右端为 `-12.50 cm`。两端横坐标差小于 20 px 时保持上一
次方向；没有方向历史则输出无效，防止符号翻转。

默认命令：

```bash
cd /home/d/2026ti/2026H/ball_yolo26s_nx
python3 tools/check_rgbd_camera.py --frames 180
python3 debug.py
python3 run.py
```

`debug.py` 默认不发串口，`run.py` 默认自动找 `/dev/ttyACM*`，其次
`/dev/ttyUSB*`。两者默认均为：

```text
--camera-backend sdk
--protocol tube-v2
--tube-length-cm 25
--tube-pose-max-age-ms 100
--positive-end image-left
--depth-width 640 --depth-height 400 --depth-fps 30
```

工程已本地安装 Orbbec SDK 2.8.6，并编译
`native/liborbbec_bridge.so`。在新系统重建依赖：

```bash
bash tools/setup_orbbec_sdk.sh
```

SDK 使用 `COLOR_FRAME_REQUIRE` 聚合，因此 60 Hz 彩色不会被 30 Hz 深度
降为 30 Hz。MJPEG 经 GStreamer `nvjpegdec` 硬件解码。深度保持厂家
640×400 原始数据，只投影采样点，不做错误的图像拉伸。

## 几何门禁

厘米输出必须同时满足：

- 当前球心位于管道轮廓扩张区；
- 管道投影长度至少 400 px；
- 有效深度纵向分箱比例至少 60%；
- 三维轴线拟合 RMS 不大于 4 mm；
- 观测管长在 22–28 cm；
- 最近管姿态不超过 100 ms。

任一门禁失败，`tube-v2` 立即发送全零无效包。管姿态线程只处理最新深度，
约 30 Hz；YOLO、厘米投影和 USB 仍按最新彩色帧约 60 Hz。球允许最多两帧
恒速预测，但过期管姿态绝不继续输出。

Debug 窗口中青色是管道轮廓，紫色是三维轴，从左至右标出 `+12.5/0/-12.5`；
同时显示球在轴上的投影、厘米位置、RGB/Depth/Pose/Inference FPS、Pitch、
深度年龄、RMS、有效分箱比例和两个置信度。现场调光时可调：

```text
--tube-white-min-gray 105
--tube-white-max-saturation 150
--tube-color-mode dark-green
--tube-green-hue-min 35
--tube-green-hue-max 95
--tube-green-min-saturation 45
--tube-green-min-value 25
--tube-green-min-excess 3
--tube-min-projection-px 400
--tube-min-depth-ratio 0.60
--tube-max-rms-mm 4
```

背景应保持暗且边缘清晰。若深度有效率长期不足 60%，不能通过降低门禁伪造
精度，应改善红外反射/曝光、相机距离或增加机械角度传感器。

Gemini 336L 在本工程的 `640×400` 深度模式下不适合 14 cm 安装距离。
25 cm 管在当前彩色内参下要满足至少 400 px 投影长度，建议先把“彩色/深度
光心到管道最高表面”的距离调整到 30–35 cm：既离开近距离深度盲区，又能
保持约 430–510 px 的管长。Debug 在三维姿态无效时仍会显示橙色
`2D REFERENCE ONLY` 刻度，便于构图，但这种刻度不能用于厘米输出；只有
紫色 `3D TUBE SCALE (cm)` 才表示深度门禁通过。

## 补采与连续片段划分

至少补采 300 张实际半管图片，空管负样本至少 20%。每次命令只创建一个
连续视频段，并在采集时明确分配 Train/Validation/Test，禁止将相邻帧随机
打散。例如：

```bash
python3 tools/collect_tube_dataset.py \
  --segment train_roll_01 --split train --mode positive --count 140
python3 tools/collect_tube_dataset.py \
  --segment val_pitch_01 --split val --mode positive --count 70
python3 tools/collect_tube_dataset.py \
  --segment test_pitch_01 --split test --mode positive --count 70
python3 tools/collect_tube_dataset.py \
  --segment train_empty_01 --split train --mode negative --count 20
```

应额外建立不同方向、`0/±10/±20°`、全管位置、运动模糊和不同光照的连续段。
正样本会用当前 Engine 生成一个初始球框，所有自动框都必须人工检查；正样本
必须恰好一个球框，负样本标签必须为空。采集器同时保存 16 位深度 PNG、
厂家标定、图像哈希和每帧管姿态诊断。

人工复核后生成融合旧数据的新数据集：

```bash
python3 prepare_tube_dataset.py
./.venv-train/bin/python train.py \
  --data dataset_tube/data.yaml \
  --model models/ball_yolo26s_best.pt
```

`prepare_tube_dataset.py` 在不足 300 张、负样本不足 20 张、缺少任一划分、
图片哈希改变、正样本不是一个框或负样本存在框时会拒绝生成。

## 五点五倾角精度验收

将球依次放在 `-12.5、-6.25、0、+6.25、+12.5 cm`，半管依次设为
`-20、-10、0、+10、+20°`。每个组合稳定采集至少 100 个推理结果：

```bash
python3 run.py --no-serial --max-seconds 3 \
  --reference-position-cm -6.25 --reference-pitch-deg 10 \
  --position-csv artifacts/calibration/m6_25_p10.csv
```

25 个组合完成后：

```bash
python3 tube_acceptance.py artifacts/calibration/*.csv
```

工具逐组合要求：不少于 100 帧、有效率至少 99%、绝对误差 P95 不大于
0.3 cm，并检查所有 25 个组合及非零基准点的符号。

连续性能验收：

```bash
python3 run.py --no-serial --max-seconds 600 \
  --metrics-json artifacts/live_tube_10min.json
```

目标为彩色采集中位数 ≥59 Hz、推理和 USB 平均 ≥58 Hz、管姿态 ≥28 Hz、
端到端 P95 ≤35 ms，且无持续队列/内存增长。当前短测已选中 640×416：
Capture 60.03 Hz、Inference 59.25 Hz、Tube pose 30.01 Hz、P95 26.29 ms；
实际白管 10 分钟、运动 5 分钟和五点精度仍必须现场完成。

## USB V2

固定 18 字节，小端：

```text
A5 5A | version=2 | payload_length=11
uint32 frame_id
uint8  status                 # 0 invalid, 1 measured, 2 predicted
int16  position_x100_cm
uint16 ball_confidence        # 0..1000
uint16 tube_confidence        # 0..1000
uint16 CRC-16/CCITT-FALSE     # covers Version through Payload
```

串口仍为 921600 8N1。串口阻塞时待发送包会被新包覆盖，不阻塞推理。
