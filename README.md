# 如何运行
## 依赖
- 官方容器镜像
- `pip install scikit-learn`

## 运行
```sh
source devel/setup.zsh

# 场景一：包裹称重与摆放
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 0

# 场景二：零件视觉分拣
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 0

# 场景三：SMT 料盘出库
rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 0
```