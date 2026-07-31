# Jetson NX 滚球控制项目

本目录是独立的 C++17 生产控制项目。它在50 Hz下接收现有 YOLO26s 视觉位置、
DMMC 管道实际角和底盘状态，输出 `tube-control-v3`，不控制车轮。

已实现的链路：

- `tube-v2/v3`、`tube-control-v3`、`tube-status-v3`、`chassis-state-v1` 固定帧、
  CRC、流式拆包、粘包/噪声重同步；
- 对需要底盘状态的通用/兼容任务提供实际加速度低延迟滤波和当前加速度前馈；
  赛题任务4/5/6明确禁用这条底盘运动输入；
- 状态 `[x, x_dot, d]` 的卡尔曼观测器、置信度动态测量噪声、3σ门限和至少
  250 ms历史；v3迟到测量在曝光时刻更新后重放；
- 所有活动任务共用带前馈、积分分离和反算抗饱和的位置外环PID；NX输出目标管角，
  MC02继续执行现有机构逆解、角度拟合和电机/管角内环；
- 管角硬限幅、角速度限制、内环角差监控和轻量安全预测；视觉丢失100～250 ms
  先进入0° HOLD软保护，超过250 ms或软边界存在
  当前/预测风险时进入锁存SAFE；底盘、DMMC超时和±11.5 cm边界也会触发安全锁存；
- Task3 使用五次连续轨迹完成 `0→+5→-5 cm`，端点采用位置、速度和实测管角
  联合稳定判据；任务4/5/6使用视觉球位置反馈的PID保持中心或指定
  位置，通用 `auto` 模式仍支持底盘阶段切换；
- Task3 根据实测静摩擦死区进行带滞回的静/动摩擦前馈，突破后平滑降补偿，
  目标死区只保留 `-0.15°` 平衡偏置；PID、观察器与绝对管角命令使用分离的角度分量；
- 每周期CSV、离线重放和位置误差、PID分量、饱和率、内环角差统计。

线协议的唯一字节级定义见 [PROTOCOL.md](PROTOCOL.md)。

## 构建和测试

构建只需要C++17、Eigen3和线程库，不再依赖OSQP或其他在线优化求解器：

```bash
cd /home/d/2026ti/2026H/ball_yolo26s_nx/nx_control
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
cd build && ctest --output-on-failure
```

`Release`构建会显式启用GCC `-O3`、LTO，并默认用`-mcpu=native`针对执行构建的
Jetson CPU优化；若在一台机器上交叉构建给不同CPU运行，可加
`-DNX_CONTROL_NATIVE_OPTIMIZATION=OFF`。

启动会打印 `Controller: cascaded PID`。外环使用观测器位置/速度，控制律为
`u=Kp·ex+Ki·I+Kd·ev+a_chassis+a_ref/lambda-Kd_dist·d/lambda`，再由
`theta=atan(u/g)`换算目标管角。积分只在误差不超过3 cm时累积，角度限幅、限速
以及Task3专项覆盖都会通过反算反馈到积分器。

无硬件烟测：

```bash
./build/ball_nx_control --config config/nx-control.conf --dry-run \
  --task center --max-seconds 2 --log logs/smoke.csv
python3 tools/analyze_log.py logs/smoke.csv
```

`--dry-run` 不打开串口或UDP，只注入健康零输入，用来检查周期、PID和日志，不是
硬件验收。

## 固定 HOLD 通信工具

`tube_hold_sender` 是独立的小工具，以50 Hz向DMMC连续发送固定的
`tube-control-v3`：

```bash
./build/tube_hold_sender --dmmc /dev/ttyACM0 --baud 921600 \
  --state-file state/tube-hold-v3.seq
```

发送字段固定为角度+20 cdeg（+0.20°）、角速度限制200 cdeg/s（2 deg/s）、TTL 200 ms、
`control_state=1`（HOLD）、`reserved=0`。首帧默认使用`flags=0x03`
（ENABLE + 清除可恢复告警），后续使用`0x01`；若不希望首帧清告警，可加
`--no-clear-faults`。

`command_id`和`source_frame_id`使用相同的严格递增值。工具在序列文件上持有
独占锁，并在发送前把一段ID预留值同步到磁盘：正常重启连续递增，异常掉电重启
可能跳号，但不会复用可能已发送的ID。序列文件不得删除、回滚、复制给另一个
同时运行的发送器；首次接入一个保留了旧命令历史的DMMC时，用
`--start-command-id LAST_ID_PLUS_ONE`指定安全起点，该参数只对新建或空文件生效。
工具在串口写入结果不确定时直接退出，不会重发同一个`command_id`。
串口打开使用独占模式；正式控制程序与本工具不能同时占用同一串口，也不能同时
锁定同一个序列文件。

无硬件检查三帧的完整字节：

```bash
./build/tube_hold_sender --dry-run --max-frames 3 \
  --start-command-id 100 --verbose
```

## 联机运行

先启动控制进程，再启动视觉进程：

```bash
./nx_control/build/ball_nx_control \
  --config nx_control/config/nx-control.conf \
  --dmmc /dev/ttyACM0 --task auto --log nx_control/logs/live.csv \
  --state-file nx_control/state/tube-control-v3.seq

python3 run.py --no-serial --protocol tube-v3 \
  --control-udp 127.0.0.1:29001
```

视觉到控制使用本机 UDP `127.0.0.1:29001`。发送为非阻塞最新帧语义；UDP丢包
不会积累旧视觉帧，控制侧按 `frame_id`、采集时刻和超时处理。DMMC使用独立的
921600-8N1双向 USB CDC。也可用 `--vision-bind`、`--vision-port`、`--baud`
覆盖。

任务参数：

- `--task 3`：赛题要求3，静止时按限速、限加速度和限 jerk 的五次轨迹执行
  `O→+5→-5 cm`；正向运动超过`+3.8 cm`开始主动反向制动，实测位置只要严格
  超过`+4.0 cm`，立即视为已经到达过`+5 cm`并切换返回段，不检查速度、实际
  管角、反馈状态、轨迹完成状态或稳定时间。若仍过冲到`+5.0 cm`外，则把反向
  减速度由`0.08 m/s²`轻度增强到`0.10 m/s²`。返程速度达到`-2.0 cm/s`后，
  先以`4°/s`直接把实测管角拉回平衡偏置；进入平衡偏置±0.2°后，该阶段只退出
  一次并恢复普通`2°/s`限速，由PID继续按位置和速度调整。返回`-5 cm`时达到
  `-4.0 cm`执行一次主动反向提前制动；最终位置误差不超过±1.0 cm、速度不超过
  5 mm/s且实际管角处于平衡偏置±0.2°内后立即完成。参考轨迹上限为`7 cm/s`、
  `10 cm/s²`和`30 cm/s³`，两段均从静止开始的保守参考总时长约`4.87 s`；
- `--task 45`：赛题要求4和5，车辆行驶阶段始终保持中心 `O`（`4`和`5`
  也是该 task 的别名）；PID只使用摄像头更新的球状态和摆杆反馈，不使用底盘
  速度、实测/参考加速度或jerk；
- `--task 6 --target-cm N`：赛题要求6，车辆行驶阶段保持指定位置 `N` cm，
  控制输入与任务4/5相同，不使用底盘运动状态；
- `--task static`：启动后执行 `0→+5→-5 cm`；
- `--task center`：保持中心；
- `--task target --target-cm N`：保持指定位置；
- `--task auto`：指定位置不变，状态随底盘启动/巡航/制动切换；
- `--wait-start`：等待 `chassis-state-v1.events bit0` 上升沿再开始；任务4/5/6
  完全不依赖底盘事件，因此不接受该选项，可直接启动或使用 `--key-start`。
- `--key-start`：在前台终端等待键盘，按`d`立即开始或从头重启一次任务，无需
  回车；该模式不响应底盘启动事件。Task3等待期持续以0°
  `HOLD`保持横梁，不关闭电机。

SAFE/Fault一旦发送即在NX侧锁存，视觉恢复不会自动回到TRACK；只能由操作员再次
按`d`重启任务。按键后的第一条非安全命令使用HOLD或TRACK并精确发送
`flags=0x03`（ENABLE + CLEAR_COMM_WARNING），后续正常帧恢复`0x01`。
bit1只是通信状态恢复标志，不是Ozone解锁码。

正式程序也使用预留号段并同步落盘的`command_id`序列文件，默认路径为
`state/tube-control-v3.seq`。可用`--state-file`和仅对新文件生效的
`--start-command-id`覆盖。串口写入结果不确定时该ID会被跳过，后续帧绝不重发
旧ID。正式程序和`tube_hold_sender`应使用各自的序列文件；切换发送者前必须先
停止当前进程。

明确的`--task 3`静止任务，以及视觉闭环的`--task 45`、`--task 6`，允许在缺少
新鲜`chassis-state-v1`时运行。其中任务4/5/6始终把底盘加速度预测域置零，收到
底盘报文也只记录而不参与控制。其他任务（包括旧`--task static`）仍要求底盘状态
新鲜且无故障。

赛题 task 示例：

```bash
# 要求3：小车静止，立即开始
./build/ball_nx_control --config config/nx-control.conf --task 3

# 要求3：等待按d执行，完成后可再次按d重复执行
./build/ball_nx_control --config config/nx-control.conf --task 3 --key-start

# 要求4/5：视觉位置闭环，立即开始保持中心
./build/ball_nx_control --config config/nx-control.conf --task 45

# 要求6：视觉位置闭环，立即开始保持 -7.3 cm
./build/ball_nx_control --config config/nx-control.conf \
  --task 6 --target-cm -7.3
```

要求6必须显式给出`--target-cm`，合法范围为`[-10, +10] cm`；要求3和要求4/5
会忽略该参数。三个赛题 task 只负责摆杆滚球控制，小车循迹、AB/整圈计时和停车
仍由底盘控制端负责。

示例 systemd 文件在 `deploy/`。其中用户、路径和串口设备是占位值，复制到系统
前必须按实际 NX 修改。

## 离线重放和统计

`ball_nx_replay` 输入CSV列固定为：

```text
time_s,vision_status,position_m,ball_conf,tube_conf,theta_actual_rad,
a_actual_m_s2,a_ref_m_s2,jerk_ref_m_s3,motion_phase
```

时间必须单调，`vision_status` 为0/1/2，`motion_phase` 为0–4。例如：

```bash
./build/ball_nx_replay capture.csv replay-output.csv config/nx-control.conf task3
python3 tools/analyze_log.py replay-output.csv --json replay-metrics.json
```

正式测试应同时保留视觉侧 `--position-csv`、控制CSV和视频。控制日志中的
`vision_age_ms`、PID各分量、饱和状态、实际管道角、底盘阶段和故障字段足以
复现观测与控制决策。Task3还记录`task3_stage`、三阶一致的`planned_*`参考、
三个`settle_*`判据及保持时间、`theta_pid/bias/friction/command_deg`、
`friction_mode/direction`和`theta_actual_deg`，可直接定位换向、突破、滚动降补偿
及最终死区。日志记录`pid_p/i/d/ff/disturbance`、未限幅/实际应用控制量、积分冻结、
积分限幅、输出饱和、`inner_angle_error_rad`和内环告警。日志还直接记录`wire_control_state`、`wire_flags`、
`dmmc_controller_state`、`safety_latched`、`safety_event_id`和
`last_stop_reason`；安全锁存/解除行会立即flush，不依赖每秒一次的终端摘要。

## 上车前必须实测的参数

默认值来自实施计划，只能作为首轮低风险参数。依次完成并写回配置：MC02空载
管角跟踪误差、`pid_kp/ki/kd`、视觉测量噪声与端到端延迟、有效
`rolling_lambda`、底盘加速度正方向和滤波时常。随后按空载角度、小球静止、
静态移动、低速直线、AB段、低速整圈、30秒整圈的顺序放开测试。

视觉实测位置在进入延迟观测器前使用
`vision_position_filter_tau_s=0.010`的一阶低通轻滤波；丢球、低置信度、
时间戳跳变或超过软丢帧时长后自动复位。未经滤波的位置仍独立用于越界保护，
进入软边界内侧1 cm的保护带时也会自动旁路滤波。

当前代码提供PID单测和离线重放；相机、DMMC及底盘实机接口
仍需按上述顺序验收，不能用软件测试结果代替真机验收。

NX的摆杆命令硬限幅为`±4.0°`，普通角速度限制为`2°/s`，Task3返程平衡阶段
临时使用`4°/s`。`tube-control-v3`在线路上仍使用厘度：
`theta_cmd_cdeg`的1 LSB为`0.01°`。

配套MC02固件应使用`default_rate_limit_deg_s=2.0`、
`minimum_rate_limit_deg_s=0.5`、`maximum_rate_limit_deg_s=4.0`和
`acceleration_limit_deg_s2=5.0`，但这些固件参数不属于本NX工程。
NX比较MC02回传的`theta_reference`与`theta_actual`：角差超过1°持续0.2秒时
请求底盘减速，超过2°持续0.5秒时锁存SAFE。该监控不改变机构逆解或拟合参数。

`HoldTarget`在位置误差小于4 mm且估计速度小于15 mm/s时进入带滞回的静止区；
位置误差超过8 mm才退出。非Task3静止区内暂停PID追踪，管道角度按2°/s限制缓慢
回到0°。Task3在更严格的2.5 mm、5 mm/s死区内撤销方向摩擦补偿并回到
`-0.15°`平衡偏置，避免反复突破静摩擦形成极限环。
