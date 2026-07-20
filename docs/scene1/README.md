# Scene 1 组员代码与文档集成说明

本目录收录场景一组员提供的文档：

- [`technical-plan-zh.md`](technical-plan-zh.md)：中文技术方案与提交材料说明。
- [`development-notes-fr.md`](development-notes-fr.md)：法文开发笔记。

对应运行代码位于：

- `scripts/scene1_task.py`
- `scripts/scene1/`
- `src/robot_api1.py`

统一入口会在 `--scene scene1` 时使用 `robot_api1`；场景二继续保留 `robot_api`，场景三继续使用 `robot_api3`。

组员原始文档中的仓库拷贝步骤和 `src/robot_api.py` 路径反映其独立仓库布局；在当前合并后的工程中，请使用上面的实际路径。场景一动作逻辑还会通过 `rospkg` 调用官方 `challenge_cup_simulator` 中的 Scene 1 控制辅助模块；该受保护依赖未被本项目修改。
