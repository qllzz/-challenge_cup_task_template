#!/usr/bin/env python3
"""
场景一：快递称重与摆放。

场景对象：
  - 快递: parcel_1、parcel_2、parcel_3、parcel_4
  - 称重区: weighing_area_0p2m_square
  - 分拣箱: sorting_box_0p4_0p3_0p3
  - 桌面: challenge_table

任务流程：
  1. 识别桌面和 4 个快递
  2. 选择目标快递，估计抓取点
  3. 手臂到位 → 打开夹爪 → 抓取 → 抬升
  4. 搬运到称重区 → 放置
  5. 再抓取 → 搬运到分拣箱 → 放置
  6. 处理下一个快递
"""

import rospy


def run_scene1(robot, arm, claw, log):
    """
    场景一任务主逻辑。

    参数:
        robot — RobotMover 实例
        arm   — ArmController 实例
        claw  — ClawController 实例
        log   — 日志函数
    """
    log("=" * 50)
    log("场景一：快递称重与摆放 — 任务开始")
    log("=" * 50)

    # ============================================================
    # 第一步：手臂切到外部控制模式（操作前必须）
    # ============================================================
    log("[STEP 1] 切换手臂到外部控制模式")
    arm.switch_to_external_control()
    rospy.sleep(1.0)

    # ============================================================
    # TODO: 在这里实现场景一的任务逻辑
    # ============================================================
    #
    # 建议的实现顺序：
    #   1. 单件快递抓取 — 识别 → 靠近 → 抓取 → 抬升
    #   2. 单件称重区放置 — 搬运 → 放置 → 松爪
    #   3. 单件分拣箱放置 — 从称重区抓回 → 放到分拣箱
    #   4. 4 件连续处理 — 循环 + 失败重试
    #   5. 多 seed 测试
    #
    # 关键接口：
    #   robot.move_forward(speed, duration)   — 靠近目标
    #   arm.go_to_joints([14个关节角度])        — 手臂到位
    #   claw.open() / claw.close()             — 夹爪开合
    #   claw.is_grabbed()                      — 判断是否抓住
    #
    # 感知接口（后续 perception_api 封装）：
    #   /cam_h/color/image_raw/compressed      头部 RGB
    #   /cam_h/depth/image_raw/compressedDepth 头部深度
    #   /lidar/points                           激光雷达点云
    #
    # 场景对象名称：
    #   parcel_1 ~ parcel_4                    快递包裹
    #   weighing_area_0p2m_square              称重区
    #   sorting_box_0p4_0p3_0p3                分拣箱
    #   challenge_table                         桌面

    # ---- 以下为占位测试代码，正式开发时请替换 ----
    log("[TEST] 前进 1 秒测试")
    robot.move_forward(0.05, duration=1.0)
    rospy.sleep(0.5)
    # -------------------------------------------

    log("场景一：任务结束")
