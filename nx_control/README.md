# Jetson NX 滚球控制项目

本目录是独立的 C++17 生产控制项目。它在50 Hz下接收现有 YOLO26s 视觉位置、
DMMC 管道实际角和底盘状态，输出 `tube-control-v3`，不控制车轮。

已实现的链路：

- `tube-v2/v3`、`tube-control-v3`、`tube-status-v3`、`chassis-state-v1` 固定帧、
  CRC、流式拆包、粘包/噪声重同步；
- 对底盘实际加速度的一阶低延迟滤波，以及基于 `a_ref + jerk_ref·t` 的预测域前馈；
- 状态 `[x, x_dot, d]` 的卡尔曼观测器、置信度动态测量噪声、3σ门限和至少
  250 ms历史；v3迟到测量在曝光时刻更新后重放；
- 状态 `[x, x_dot, d, u_actual]`、执行机构一阶动态、增量输入决策的30步 MPC；
  角度、角速度硬约束与球位置软约束均直接进入QP；
- OSQP 0.6.x生产后端，以及无需外部求解器、仅供构建和回归测试的稠密ADMM后端；
- 单次QP失败使用上一最优序列移位，连续失败切换前馈+状态反馈备用控制，持续失败
  请求停车；视觉、底盘、DMMC超时和±11.5 cm边界均进入安全标志；
- 静态 `0→+5→-5 cm`、中心保持、指定位置保持、随底盘阶段切换的任务管理；
- 每周期 CSV（含预测位置序列）、离线重放和误差/饱和/求解耗时统计。

线协议的唯一字节级定义见 [PROTOCOL.md](PROTOCOL.md)。

## 构建和测试

本机无OSQP时可先验证全部逻辑：

```bash
cd /home/d/2026ti/2026H/ball_yolo26s_nx/nx_control
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
cd build && ctest --output-on-failure
```

比赛部署必须安装 OSQP 0.6.x 的头文件和共享库，并明确启用、强制检查：

```bash
cmake -S . -B build-osqp -DCMAKE_BUILD_TYPE=Release \
  -DNX_CONTROL_USE_OSQP=ON
cmake --build build-osqp -j2
sed -i 's/require_osqp=false/require_osqp=true/' config/nx-control.conf
```

若系统找不到OSQP，可另外传入 `-DOSQP_INCLUDE_DIR=... -DOSQP_LIBRARY=...`。
项目按OSQP 0.6.x C API编译；不要在未完成API适配时直接替换成1.x。

无硬件烟测：

```bash
./build/ball_nx_control --config config/nx-control.conf --dry-run \
  --task center --max-seconds 2 --log logs/smoke.csv
python3 tools/analyze_log.py logs/smoke.csv
```

`--dry-run` 不打开串口或UDP，只注入健康零输入，用来检查周期、MPC和日志，不是
硬件验收。

## 联机运行

先启动控制进程，再启动视觉进程：

```bash
./nx_control/build/ball_nx_control \
  --config nx_control/config/nx-control.conf \
  --dmmc /dev/ttyACM0 --task auto --log nx_control/logs/live.csv

python3 run.py --no-serial --protocol tube-v3 \
  --control-udp 127.0.0.1:29001
```

视觉到控制使用本机 UDP `127.0.0.1:29001`。发送为非阻塞最新帧语义；UDP丢包
不会积累旧视觉帧，控制侧按 `frame_id`、采集时刻和超时处理。DMMC使用独立的
921600-8N1双向 USB CDC。也可用 `--vision-bind`、`--vision-port`、`--baud`
覆盖。

任务参数：

- `--task static`：启动后执行 `0→+5→-5 cm`；
- `--task center`：保持中心；
- `--task target --target-cm N`：保持指定位置；
- `--task auto`：指定位置不变，状态随底盘启动/巡航/制动切换；
- `--wait-start`：等待 `chassis-state-v1.events bit0` 上升沿再开始。

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
./build/ball_nx_replay capture.csv replay-output.csv config/nx-control.conf
python3 tools/analyze_log.py replay-output.csv --json replay-metrics.json
```

正式测试应同时保留视觉侧 `--position-csv`、控制CSV和视频。控制日志中的
`vision_age_ms`、`mpc_ms`、`prediction_m`、实际管道角、底盘阶段和故障字段足以
复现观测与控制决策。

## 上车前必须实测的参数

默认值来自实施计划，只能作为首轮低风险参数。依次完成并写回配置：执行机构
`actuator_tau_s/actuator_delay_s`，视觉测量噪声与端到端延迟，有效
`rolling_lambda`，底盘加速度正方向和滤波时常。随后按空载角度、小球静止、
静态移动、低速直线、AB段、低速整圈、30秒整圈的顺序放开测试。

当前代码已通过软件构建和仿真单测；相机、DMMC、底盘实机接口及OSQP在本工作区
没有可用硬件/库，不能用软件测试结果代替真机验收。
