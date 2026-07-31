# NX 控制通信协议

NX 当前使用两种线格式：

- 视觉 UDP 使用 NX 视觉帧格式；
- DMMC02 串口使用现有 MC02 固件帧格式。

两者不可混用。`src/main.cpp` 分别使用 `StreamParser` 和
`Mc02StreamParser` 解析这两条链路。

DMMC02本地管角/电机控制环为1 kHz，但NX命令和DMMC02状态仍以50 Hz传输。
本地控制周期升级不改变本文件定义的任何字段偏移、长度、TTL、`command_id`或CRC。

## 视觉 UDP 帧格式

所有多字节字段均为小端：

| 偏移 | 长度 | 字段 |
|---:|---:|---|
| 0 | 2 | Magic `A5 5A` |
| 2 | 1 | 类型/版本 |
| 3 | 2 | Payload 长度 |
| 5 | N | Payload |
| 5+N | 2 | CRC-16/CCITT-FALSE |

CRC 覆盖类型、长度和 Payload，不覆盖 Magic。标准向量
`123456789` 必须得到 `0x29B1`。

### 视觉 `tube-v2` / `tube-v3`

`tube-v2` 类型为 `0x02`，Payload 11字节，总长18字节。
`tube-v3` 类型为 `0x03`，在相同11字节之后追加采集时刻，
Payload 15字节，总长22字节。

| Payload偏移 | 长度 | 类型 | 字段/单位 |
|---:|---:|---|---|
| 0 | 4 | uint32 | frame_id |
| 4 | 1 | uint8 | status：丢失/实测/短时预测 |
| 5 | 2 | int16 | position，m×10000 |
| 7 | 2 | uint16 | ball_confidence，0～1000 |
| 9 | 2 | uint16 | tube_confidence，0～1000 |
| 11 | 4 | uint32 | capture_time_ms，仅v3 |

状态为丢失时，位置和两个置信度必须为0。

## MC02 串口帧格式

所有多字节字段均为小端：

| 偏移 | 长度 | 字段 |
|---:|---:|---|
| 0 | 2 | Magic `A5 5A` |
| 2 | 1 | 消息类型 |
| 3 | 1 | Payload 长度 |
| 4 | N | Payload |
| 4+N | 2 | CRC-16/CCITT-FALSE，低字节在前 |

CRC 覆盖 Magic、类型、长度和 Payload，即除末尾CRC外的整个帧。
MC02接受的最大Payload长度为56字节。

### NX→MC02 `tube-control-v3`

类型为 `0x80`，Payload 22字节，总长28字节：

| Payload偏移 | 长度 | 类型 | 字段/单位 |
|---:|---:|---|---|
| 0 | 4 | uint32 | command_id |
| 4 | 4 | uint32 | source_frame_id |
| 8 | 4 | uint32 | nx_time_ms |
| 12 | 2 | int16 | theta_cmd，0.01° |
| 14 | 2 | uint16 | theta_rate_limit，0.01°/s |
| 16 | 2 | uint16 | ttl_ms |
| 18 | 1 | uint8 | MC02控制状态 |
| 19 | 1 | uint8 | flags |
| 20 | 2 | uint16 | 保留，发送0 |

NX任务状态在线路上按下表映射：

| NX状态 | MC02状态 |
|---|---|
| Idle | Disabled (`0`) |
| StandbyHold | Hold (`1`) |
| StaticMove、HoldCenter、HoldTarget | Track (`2`) |
| VehicleAccel、VehicleCruise、VehicleDecel | Track (`2`) |
| Safe | Safe (`3`) |
| Fault | Fault (`4`) |

`flags`：bit0机构使能、bit1清除通信可恢复告警（`CLEAR_COMM_WARNING`）、
bit2建议底盘减速、
bit3建议底盘停车。

Task3等待启动按键时使用`StandbyHold`，固定下发0°并保持机构使能。
固定HOLD通信工具也在线路上直接发送`control_state=1`。
bit1是通信状态恢复标志，不是Ozone解锁码。操作员按`d`重启任务后的第一条
非安全命令使用`flags=0x03`，后续正常命令恢复`0x01`。

固定测试向量：

```text
A5 5A 80 16 01 00 00 00 00 00 00 00 00 00 00 00
00 00 C8 00 3C 00 04 08 00 00 6B 58
```

其中角速度限制为200 cdeg/s（2°/s），CRC数值为 `0x586B`，线上小端字节为
`6B 58`。

### MC02→NX `tube-status-v3`

类型为 `0x90`，Payload 52字节，总长58字节：

| Payload偏移 | 长度 | 类型 | 字段/单位 |
|---:|---:|---|---|
| 0 | 4 | uint32 | status_id |
| 4 | 4 | uint32 | MC02本地时间，ms |
| 8 | 4 | uint32 | 回显command_id |
| 12/14/16 | 各2 | int16 | target/reference/actual，0.01° |
| 18 | 4 | int32 | motor_position，mrad |
| 22 | 4 | int32 | motor_velocity，mrad/s |
| 26 | 2 | int16 | motor_torque，mN·m |
| 28 | 1 | uint8 | 控制器状态 |
| 29 | 1 | uint8 | 电机状态 |
| 30 | 4 | uint32 | fault_flags |
| 34 | 2 | uint16 | CAN反馈年龄，ms |
| 36 | 2 | uint16 | USB控制命令年龄，ms |
| 38 | 2 | uint16 | 有效USB帧数 |
| 40 | 2 | uint16 | USB CRC错误数 |
| 42 | 2 | uint16 | USB序列错误数 |
| 44 | 2 | uint16 | CAN反馈帧数 |
| 46 | 2 | uint16 | 1 kHz本地控制周期超时数（字段位置和宽度不变） |
| 48 | 2 | uint16 | USB发送丢帧数 |
| 50 | 2 | uint16 | 底盘安全请求标志 |

### MC02→NX `chassis-state-v1`

类型为 `0x91`，Payload 40字节，总长46字节：

| Payload偏移 | 长度 | 类型 | 字段/单位 |
|---:|---:|---|---|
| 0 | 4 | uint32 | sequence |
| 4 | 4 | uint32 | 底盘源时间，ms |
| 8 | 4 | uint32 | MC02接收时间，ms |
| 12 | 4 | int32 | v_ref，mm/s |
| 16 | 4 | int32 | a_ref，mm/s² |
| 20 | 4 | int32 | jerk_ref，mm/s³ |
| 24 | 4 | int32 | v_actual，mm/s |
| 28 | 4 | int32 | a_actual，mm/s² |
| 32 | 1 | uint8 | motion_phase |
| 33 | 1 | uint8 | track_segment |
| 34 | 1 | uint8 | track_quality，NX归一化为0～1 |
| 35 | 1 | uint8 | events |
| 36 | 2 | uint16 | ttl_ms |
| 38 | 2 | uint16 | fault_flags |

MC02当前不提供 `track_error` 和 `yaw_rate`，NX解码后将它们保留为0。
