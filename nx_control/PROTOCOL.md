# NX 控制通信协议

所有多字节字段均为小端。帧结构统一为：

| 偏移 | 长度 | 字段 |
|---:|---:|---|
| 0 | 2 | Magic `A5 5A` |
| 2 | 1 | 类型/版本 |
| 3 | 2 | Payload 长度 |
| 5 | N | Payload |
| 5+N | 2 | CRC-16/CCITT-FALSE |

CRC 覆盖类型、长度和 Payload，不覆盖 Magic。标准向量 `123456789` 必须得到
`0x29B1`。接收器按 Magic 重同步，拒绝超过128字节的 Payload、错误长度和错误
CRC。

## 视觉 `tube-v2` / `tube-v3`

`tube-v2` 类型为 `0x02`，Payload 11字节，总长18字节。`tube-v3` 类型为
`0x03`，在相同11字节之后追加采集时刻，总长22字节：

| Payload偏移 | 长度 | 类型 | 字段/单位 |
|---:|---:|---|---|
| 0 | 4 | uint32 | frame_id |
| 4 | 1 | uint8 | status：0丢失、1实测、2短时预测 |
| 5 | 2 | int16 | position，cm×100 |
| 7 | 2 | uint16 | ball_confidence，0–1000 |
| 9 | 2 | uint16 | tube_confidence，0–1000 |
| 11 | 4 | uint32 | capture_time_ms，仅v3，NX `CLOCK_MONOTONIC` 低32位 |

状态0时位置和两个置信度必须为0，采集时刻仍保留。C++端处理32位回绕，并把
迟到测量放回曝光时刻更新后重放到当前控制时刻。

## NX→DMMC `tube-control-v3`

类型 `0x80`，Payload 21字节，总长28字节：

| Payload偏移 | 长度 | 类型 | 字段/单位 |
|---:|---:|---|---|
| 0 | 4 | uint32 | command_id |
| 4 | 4 | uint32 | source_frame_id |
| 8 | 4 | uint32 | nx_time_ms |
| 12 | 2 | int16 | theta_cmd，0.01° |
| 14 | 2 | int16 | theta_rate_limit，0.01°/s |
| 16 | 2 | uint16 | ttl_ms |
| 18 | 1 | uint8 | control_state |
| 19 | 1 | uint8 | flags |
| 20 | 1 | uint8 | 保留，发送0 |

`flags`：bit0机构使能，bit1清除可恢复告警，bit2建议底盘减速，bit3建议底盘
停车。`control_state` 的0–8依次为 IDLE、STATIC_MOVE、HOLD_CENTER、
HOLD_TARGET、VEHICLE_ACCEL、VEHICLE_CRUISE、VEHICLE_DECEL、SAFE、FAULT。

## DMMC→NX `tube-status-v3`

类型 `0x90`，Payload 32字节，总长39字节：

| Payload偏移 | 长度 | 类型 | 字段/单位 |
|---:|---:|---|---|
| 0 | 4 | uint32 | sequence |
| 4 | 4 | uint32 | dmmc_time_ms |
| 8/10/12 | 各2 | int16 | target/reference/actual，0.01° |
| 14 | 4 | int32 | motor_position，mrad |
| 18 | 2 | int16 | motor_velocity，0.01 rad/s |
| 20 | 2 | int16 | motor_torque，mN·m |
| 22 | 2 | uint16 | faults |
| 24 | 2 | uint16 | can_age_ms |
| 26 | 2 | uint16 | usb_crc_errors |
| 28 | 2 | uint16 | control_age_ms |
| 30 | 1 | uint8 | DMMC状态 |
| 31 | 1 | uint8 | 状态标志 |

## DMMC→NX `chassis-state-v1`

类型 `0x91`，Payload 32字节，总长39字节：

| Payload偏移 | 长度 | 类型 | 字段/单位 |
|---:|---:|---|---|
| 0 | 4 | uint32 | sequence |
| 4 | 4 | uint32 | chassis_time_ms |
| 8 | 2 | int16 | v_ref，mm/s |
| 10 | 2 | int16 | a_ref，mm/s² |
| 12 | 2 | int16 | jerk_ref，mm/s³ |
| 14 | 2 | int16 | v_actual，mm/s |
| 16 | 2 | int16 | a_actual，mm/s² |
| 18 | 2 | int16 | track_error，0.1 mm |
| 20 | 2 | uint16 | track_quality，0–1000 |
| 22 | 1 | uint8 | motion_phase：停止/启动/巡航/弯道/制动=0–4 |
| 23 | 1 | uint8 | track_segment：未知/AB/BC/CD/DA=0–4 |
| 24 | 2 | uint16 | events；bit0为开始事件 |
| 26 | 2 | uint16 | faults |
| 28 | 2 | uint16 | ttl_ms |
| 30 | 2 | int16 | yaw_rate，mrad/s |

本文件是 NX 与 DMMC 两端的字节级契约；DMMC 工程实现时应直接用这里的固定
长度和缩放，避免依赖 C 结构体对齐。
