# 代码结构说明

## `scripts/` — 在这里写任务逻辑

```
scripts/
├── challenge_task.py   ← 【不用改】主入口，负责启动仿真 + 分发场景
├── scene1_task.py      ← 【你要改】场景一：快递称重与摆放
├── scene2_task.py      ← 【你要改】场景二：零件分拣归档
└── scene3_task.py      ← 【你要改】场景三：SMT 料盘出库
```

- **`challenge_task.py`**：统一入口，处理 `--scene scene1/2/3 --seed 3` 参数、启动仿真器、创建控制器对象，然后根据场景参数调用对应的 `run_sceneX()`。不需要改它。

- **`scene1_task.py`**：场景一的任务逻辑写在 `run_scene1(robot, arm, claw, log)` 函数里。三个参数是封装好的控制器（走路、手臂、夹爪），`log` 是打印函数（用法同 `print`，带 `>>>` 前缀）。

- **`scene2_task.py`**、**`scene3_task.py`**：同上，各自写各自场景的逻辑。三人可以同时改不同文件，不会冲突。

### 场景函数长这样

```python
def run_scene1(robot, arm, claw, log):
    arm.switch_to_external_control()   # 操作手臂前必须调用
    # ↓ 你的任务逻辑写在这里 ↓
    robot.move_forward(0.1, duration=2.0)  # 前进 2 秒
    arm.go_ready()                         # 手臂前伸
    claw.close()                           # 夹爪闭合
```

---

## `src/` — 封装好的机器人 API（直接用，不用看内部实现）

```
src/
└── robot_api.py   ← 三个控制器类
```

### RobotMover — 走路

| 方法 | 效果 | 示例 |
|------|------|------|
| `move_forward(速度, duration=秒)` | 前进，到时自动停 | `robot.move_forward(0.1, duration=2.0)` |
| `move_backward(速度, duration=秒)` | 后退 | `robot.move_backward(0.05, duration=1.0)` |
| `move_left(速度, duration=秒)` | 左移（双足危险，速度要小） | `robot.move_left(0.05, duration=1.0)` |
| `move_right(速度, duration=秒)` | 右移 | `robot.move_right(0.05, duration=1.0)` |
| `turn_left(角速度, duration=秒)` | 左转，单位 rad/s | `robot.turn_left(0.3, duration=1.5)` |
| `turn_right(角速度, duration=秒)` | 右转 | `robot.turn_right(0.3, duration=1.5)` |
| `move(forward, left, turn, duration)` | 组合移动 | `robot.move(forward=0.1, turn=0.2, duration=2.0)` |
| `stop()` | 急停，发零指令 0.5 秒 | `robot.stop()` |

`duration` 不传的话只发一条指令（机器人会一直走），传了会在时间到后自动停。

### ArmController — 手臂

| 方法 | 效果 | 示例 |
|------|------|------|
| `switch_to_external_control()` | **操作前必须调用！** 切到外部控制模式 | `arm.switch_to_external_control()` |
| `go_to_joints([14个角度])` | 发送 14 个关节角度（前 7 左臂，后 7 右臂） | `arm.go_to_joints([30, -10, 0, -60, 0, 0, 0] * 2)` |
| `go_home()` | 归零，自然下垂 | `arm.go_home()` |
| `go_ready()` | 双臂前伸准备抓取（肩 30° 肘 60°） | `arm.go_ready()` |
| `left_arm_to([7个角度])` | 只动左臂 | `arm.left_arm_to([40, -15, 0, -60, 0, 0, 0])` |
| `right_arm_to([7个角度])` | 只动右臂 | `arm.right_arm_to([40, 15, 0, -60, 0, 0, 0])` |

7 个关节顺序：`[肩膀俯仰, 肩膀侧摆, 肩膀旋转, 肘部, 手腕旋转, 手腕俯仰, 手腕侧摆]`，单位是**度**。

### ClawController — 夹爪

| 方法 | 效果 | 示例 |
|------|------|------|
| `open()` | 双手张开（默认 10%） | `claw.open()` |
| `close()` | 双手闭合（默认 90%） | `claw.close()` |
| `set_position(左%, 右%)` | 分别设定开合，0=全开 100=全闭 | `claw.set_position(80, 20)` |
| `left_open()` / `left_close()` | 只动左手 | `claw.left_close()` |
| `right_open()` / `right_close()` | 只动右手 | `claw.right_open()` |
| `is_grabbed()` | 返回 `True` 表示抓住东西了 | `if claw.is_grabbed(): ...` |
| `wait_until_done(timeout=秒)` | 阻塞等待夹爪运动完 | `claw.wait_until_done(timeout=3.0)` |

夹爪状态值：`-1` 错误、`0` 未知、`1` 移动中、`2` 已到位、`3` 已抓住物体。

---

## 开发流程

```bash
# 启动仿真（场景一为例）
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3

# 改 scene1_task.py → 保存 → 重新跑上面那行
```

改哪个场景就对应改哪个 `sceneX_task.py`，不用碰主入口。

