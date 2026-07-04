#!/usr/bin/env python3
"""
场景二：零件分拣归档。

场景对象：
  - A 类: part_type_a_1、part_type_a_2  →  sorting_bin_a
  - B 类: part_type_b_1、part_type_b_2  →  sorting_bin_b
  - C 类: part_type_c_1、part_type_c_2  →  sorting_bin_c（螺丝刀模型）

任务流程：
  1. 识别桌面上 6 个零件
  2. 判断零件类别 A/B/C
  3. 估计零件中心和主方向，生成抓取点
  4. 抓取零件 → 搬运到对应收纳盒 → 放置
  5. 处理下一个零件
"""

import rospy


def run_scene2(robot, arm, claw, log):
    """
    场景二任务主逻辑。

    参数:
        robot — RobotMover 实例
        arm   — ArmController 实例
        claw  — ClawController 实例
        log   — 日志函数
    """
    log("=" * 50)
    log("场景二：零件分拣归档 — 任务开始")
    log("=" * 50)

    # ============================================================
    # 第一步：手臂切到外部控制模式
    # ============================================================
    log("[STEP 1] 切换手臂到外部控制模式")
    arm.switch_to_external_control()
    rospy.sleep(1.0)

    # ============================================================
    # TODO: 在这里实现场景二的任务逻辑
    # ============================================================
    #
    # 建议的实现顺序：
    #   1. 先分类识别 6 个零件（YOLO/颜色/形状特征）
    #   2. 实现 A 类零件分拣 → 放到 sorting_bin_a
    #   3. 扩展 B 类 → sorting_bin_b
    #   4. 单独处理 C 类螺丝刀抓取姿态（长轴方向夹取）
    #   5. 完成 6 个零件连续分拣
    #
    # 关键接口：
    #   robot.move_forward(speed, duration)
    #   arm.go_to_joints([14个关节角度])
    #   arm.left_arm_to([7]) / arm.right_arm_to([7])
    #   claw.open() / claw.close()
    #   claw.is_grabbed()
    #
    # 场景对象名称：
    #   part_type_a_1, part_type_a_2  →  sorting_bin_a
    #   part_type_b_1, part_type_b_2  →  sorting_bin_b
    #   part_type_c_1, part_type_c_2  →  sorting_bin_c（螺丝刀）

    # ---- 以下为占位测试代码，正式开发时请替换 ----
    log("[TEST] 手臂 go_ready 前伸")
    arm.go_ready()
    rospy.sleep(2.0)
    arm.go_home()
    rospy.sleep(1.0)
    # -------------------------------------------

    log("场景二：任务结束")
