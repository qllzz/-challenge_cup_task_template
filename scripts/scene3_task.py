#!/usr/bin/env python3
"""
场景三：SMT 料盘出库。

场景对象：
  - SMT 货架: smt_rack（上下两层）
  - 料盘: smt_tray_1 ~ smt_tray_5
  - 出库区: sorting_box_0p4_0p3_0p3

任务流程：
  1. 识别 smt_rack 和目标料盘
  2. 判断料盘所在层级（上层 10 分，下层 20 分）
  3. 估计料盘前缘、中心、拉出方向
  4. 移动到货架前预操作位
  5. 手臂进入预托取位姿 → 抓住/托住料盘前缘
  6. 沿货架外法线低速拉出 → 抬升到安全高度
  7. 搬运到出库区 → 放置
"""

import rospy


def run_scene3(robot, arm, claw, log):
    """
    场景三任务主逻辑。

    参数:
        robot — RobotMover 实例
        arm   — ArmController 实例
        claw  — ClawController 实例
        log   — 日志函数
    """
    log("=" * 50)
    log("场景三：SMT 料盘出库 — 任务开始")
    log("=" * 50)

    # ============================================================
    # 第一步：手臂切到外部控制模式
    # ============================================================
    log("[STEP 1] 切换手臂到外部控制模式")
    arm.switch_to_external_control()
    rospy.sleep(1.0)

    # ============================================================
    # TODO: 在这里实现场景三的任务逻辑
    # ============================================================
    #
    # 建议的实现顺序：
    #   1. 先完成上层单个料盘出库
    #   2. 再完成下层单个料盘出库（下层更难，分值更高）
    #   3. 加入货架碰撞安全边界
    #   4. 完成两个目标料盘连续出库
    #   5. 多 seed 测试
    #
    # 关键接口：
    #   robot.move_forward(speed, duration)      — 移动到货架前
    #   robot.move(forward=..., left=..., ...)    — 微调对准
    #   arm.go_to_joints([14个关节角度])           — 手臂到位
    #   claw.open() / claw.close()                — 夹爪控制
    #   claw.is_grabbed()                         — 判断是否抓住/托住
    #
    # 场景对象名称：
    #   smt_rack                                   SMT 货架
    #   smt_tray_1 ~ smt_tray_5                   可抓取料盘
    #   sorting_box_0p4_0p3_0p3                   出库箱

    # ---- 以下为占位测试代码，正式开发时请替换 ----
    log("[TEST] 夹爪开合测试")
    claw.close()
    rospy.sleep(1.0)
    claw.open()
    rospy.sleep(1.0)
    # -------------------------------------------

    log("场景三：任务结束")
