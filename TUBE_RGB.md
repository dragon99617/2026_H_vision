# Gemini 336L 纯 RGB 墨绿色/白色管道位置

## 原理与启动

本模式不读取深度、不做人工标定，也不依赖相机安装高度。每帧从暗背景中提取
完整管道轮廓，拟合二维长轴及左右端点；左端固定为 `+12.50 cm`，右端固定为
`-12.50 cm`。YOLO 球心正交投影到该轴，按端点之间的比例线性换算：

```text
t = dot(ball - right, left - right) / |left - right|²
position_cm = clamp(t, 0, 1) × 25 - 12.5
```

调试和正式运行分别使用独立入口：

```bash
cd /home/d/2026ti/2026H/ball_yolo26s_nx
python3 debug_rgb.py
python3 run_rgb.py
```

当前墨绿色管道建议明确使用：

```bash
python3 debug_rgb.py --tube-color-mode dark-green
python3 run_rgb.py --tube-color-mode dark-green
```

默认 `--tube-color-mode auto` 会优先检测墨绿色长轮廓，找不到后再兼容原白管。
墨绿色分割同时检查 HSV 色相、饱和度、亮度和绿色通道优势，并把与绿色区域
邻接的白色反光合并，避免高光把轮廓切断。

两个入口都通过 Orbbec SDK 只开启 `1280×800@60` 彩色 MJPEG，深度流保持
关闭。Debug 中绿色轴为有效的 `RGB CONTOUR SCALE (cm)`，黄色刻度间隔
2.5 cm，画面从左至右显示 `+12.5/+6.25/0/-6.25/-12.5`、球心投影点、厘米位置、管长
像素数、轮廓置信度、曝光和性能参数。球或完整轮廓任一无效时，`tube-v2`
立即发送全零无效包。

## 曝光

彩色自动曝光默认关闭。程序读取 Gemini 336L 的曝光/增益范围，把曝光设为
SDK 出厂默认值的 30%，增益固定为 SDK 默认值，从而降低当前过曝并避免滚动
过程中亮度和快门变化。启动日志和 Debug 顶栏显示实际值：

```text
AE=off exp=... gain=...
```

现场可直接覆盖：

```bash
# 仍偏亮：降到出厂曝光的 20%
python3 debug_rgb.py --color-exposure-scale 0.20

# 太暗：升到 45%
python3 debug_rgb.py --color-exposure-scale 0.45

# 使用日志中相同单位给出固定值
python3 debug_rgb.py --color-exposure 5000 --color-gain 16

# 光照持续变化时才恢复自动曝光
python3 debug_rgb.py --color-auto-exposure
```

确认参数后，`run_rgb.py` 使用同样参数启动。曝光值会按相机实际范围和步长
自动限幅；不支持的控制项会在日志中明确报错。

## 门禁、协议与精度

- 管道投影长度默认至少 400 px，背景应与管道颜色有明显差异且整根管完整可见。
- 球心必须位于管道轮廓扩张区；轮廓端点使用指数平滑，默认
  `--rgb-tube-alpha 0.35`。
- 正方向默认取图像横坐标较小的端点（画面左端），位置限制在
  `[-12.50,+12.50] cm`。
- 如需兼容旧方向，可显式传入 `--positive-end image-right`。
- USB 沿用固定18字节 `tube-v2`；`tube_confidence` 在本模式表示二维轮廓
  置信度，而不是深度拟合置信度。

墨绿色默认门限采用 OpenCV HSV 标度：

```text
--tube-green-hue-min 35
--tube-green-hue-max 95
--tube-green-min-saturation 45
--tube-green-min-value 25
--tube-green-min-excess 3
```

若现场画面中的管道过暗，先提高曝光；仍缺失时可把
`--tube-green-min-value` 降至 `15–20`。若背景绿色物体被误选，应缩窄色相
范围或提高 `--tube-green-min-excess`，不能直接取消长度和长宽比门禁。

纯 RGB 只能测量图像上的比例。管道前后倾斜会带来透视误差，因此
`±0.5 cm` 是现场验收目标，不是未经实测的保证。无需标定，但应把球放在
`-12.5/-6.25/0/+6.25/+12.5 cm`，覆盖实际最小、中间和最大倾角，每组采集
至少100帧；要求绝对误差 P95 不大于0.5 cm且无正负翻转。若某个倾角不通过，
需提高相机高度、减小前后倾角或改用 RGB-D 版，不能通过修改管长伪造精度。

当前NX与336L的8秒现场短测结果：彩色采集中位数 `60.41 FPS`、轮廓
`59.91 FPS`、推理平均 `59.92 FPS`、端到端延迟 P95 `29.57 ms`、深度帧
为0；轮廓与小球在全部短测帧中均有效。该结果只证明实时链路，不替代上述
五点、多倾角的 `±0.5 cm` 精度验收。
