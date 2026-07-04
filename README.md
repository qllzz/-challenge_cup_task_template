程式碼結構說明
scripts/ — 你在這裡寫任務邏輯

scripts/
├── challenge_task.py   ← 【不用改】主入口，負責啟動仿真 + 分發場景
├── scene1_task.py      ← 【你要改】場景一：快遞稱重與擺放
├── scene2_task.py      ← 【你要改】場景二：零件分揀歸檔
└── scene3_task.py      ← 【你要改】場景三：SMT 料盤出庫
challenge_task.py：統一入口，處理 --scene scene1/2/3 --seed 3 參數、啟動仿真器、建立控制器物件，然後根據場景參數調用對應的 run_sceneX()。不需要改它。

scene1_task.py：場景一的任務邏輯寫在 run_scene1(robot, arm, claw, log) 函數裡。三個參數是封裝好的控制器（走路、手臂、夾爪），log 是打印函數（用法同 print，但帶 >>> 前綴）。

scene2_task.py、scene3_task.py：同上，各自寫各自場景的邏輯。三人可以同時改不同檔案，不會衝突。

場景函數長這樣

def run_scene1(robot, arm, claw, log):
    arm.switch_to_external_control()   # 操作手臂前必須調用
    # ↓ 你的任務邏輯寫在這裡 ↓
    robot.move_forward(0.1, duration=2.0)  # 前進 2 秒
    arm.go_ready()                         # 手臂前伸
    claw.close()                           # 夾爪閉合
src/ — 封裝好的機器人 API（直接用，不用看內部實現）

src/
└── robot_api.py   ← 三個控制器類別
RobotMover — 走路
方法	效果	範例
move_forward(速度, duration=秒)	前進，到時自動停	robot.move_forward(0.1, duration=2.0)
move_backward(速度, duration=秒)	後退	robot.move_backward(0.05, duration=1.0)
move_left(速度, duration=秒)	左移（雙足危險，速度要小）	robot.move_left(0.05, duration=1.0)
move_right(速度, duration=秒)	右移	robot.move_right(0.05, duration=1.0)
turn_left(角速度, duration=秒)	左轉，單位 rad/s	robot.turn_left(0.3, duration=1.5)
turn_right(角速度, duration=秒)	右轉	robot.turn_right(0.3, duration=1.5)
move(forward, left, turn, duration)	組合移動	robot.move(forward=0.1, turn=0.2, duration=2.0)
stop()	急停，發零指令 0.5 秒	robot.stop()
duration 不傳的話只發一條指令（機器人會一直走），傳了會在時間到後自動停。

ArmController — 手臂
方法	效果	範例
switch_to_external_control()	操作前必須調用！ 切到外部控制模式	arm.switch_to_external_control()
go_to_joints([14個角度])	發送 14 個關節角度（前 7 左臂，後 7 右臂）	arm.go_to_joints([30, -10, 0, -60, 0, 0, 0] * 2)
go_home()	歸零，自然下垂	arm.go_home()
go_ready()	雙臂前伸準備抓取（肩 30° 肘 60°）	arm.go_ready()
left_arm_to([7個角度])	只動左臂	arm.left_arm_to([40, -15, 0, -60, 0, 0, 0])
right_arm_to([7個角度])	只動右臂	arm.right_arm_to([40, 15, 0, -60, 0, 0, 0])
7 個關節順序：[肩膀俯仰, 肩膀側擺, 肩膀旋轉, 肘部, 手腕旋轉, 手腕俯仰, 手腕側擺]，單位是度。

ClawController — 夾爪
方法	效果	範例
open()	雙手張開（預設 10%）	claw.open()
close()	雙手閉合（預設 90%）	claw.close()
set_position(左%, 右%)	分別設定開合，0=全開 100=全閉	claw.set_position(80, 20)
left_open() / left_close()	只動左手	claw.left_close()
right_open() / right_close()	只動右手	claw.right_open()
is_grabbed()	回傳 True 表示抓住東西了	if claw.is_grabbed(): ...
wait_until_done(timeout=秒)	阻塞等待夾爪運動完	claw.wait_until_done(timeout=3.0)
夾爪狀態值：-1 錯誤、0 未知、1 移動中、2 已到位、3 已抓住物體。

開發流程

# 啟動仿真（場景一為例）
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3

# 改 scene1_task.py → 儲存 → 重新跑上面那行
改哪個場景就對應改哪個 sceneX_task.py，不用碰主入口。
