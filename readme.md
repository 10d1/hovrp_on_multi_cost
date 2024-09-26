# 基于蚁群算法的两用混合计费方式的路径规划问题
![PyPI - Implementation](https://img.shields.io/pypi/implementation/NetworkX)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/NetworkX)
![Static Badge](https://img.shields.io/badge/Package-NetworkX-blue)
![Static Badge](https://img.shields.io/badge/AntColony-red)
![Static Badge](https://img.shields.io/badge/VehicleRouting-green)
---
本项目包含如下功能：

- 测试数据生成: `datagen/generate_locations.py`
  - 用以生成带有n个节点的随机测试集，并模拟零担运费
- 求解问题：`optimizer/solve_prob.py`
  - 调用蚁群算法求解路径规划问题，根据进行每一轮次的信息素迭代，保存结果并画图。
- 测试验证：`optimizer/brute_force.py`
  - 对于小规模的数据集，用暴力穷举算法生成所有路径的排列组合用以确切得知最优解的位置。
  - 此结果用于验证蚁群算法的求解效果。
- 参数调试： `optimizer/param_tune.py`
  - 生成多个参数组合，对于目标数据集运行多次蚁群算法求解，最后输出各个参数的效果。

------
## 关键参数
### 问题生成
    - data_path：生成结果存储的目录，对应的目录下应包含一个名为 cost_funcs的文件夹，用以存储每个节点本身的FTL和LTL费用函数图像
    - ftl_cost： 整车物流的分段函数参数，应包含每种车型的负载上限，车辆的起运费用，以及车辆行驶的单位里程费用
    - demands_range: 需求分布的范围
    - n_nodes：所生成问题的需求节点的个数，其中不包括源节点和虚拟节点
    - resolution： 所生成图像的分辨率， 以多少像素代表1公里

### 蚁群算法
    - DATA_PATH：要读入的数据集的存储位置
    - OUTPUT_PATH：结果输出的存储位置，如果需要画图，将在OUTPUT_PATH下自动建立一个img文件夹，用以存储每轮次的信息素和解的图像
    - ALPHA：信息素的重要程度 （0~1）
    - BETA：期望信息素浓度（0~1）
    - RHO：信息素浓度衰减系数（0~1）
    - MAX_LOOP：求解时允许的迭代次数
    - NO_IMPRO_THRESHOLD： 当连续数轮迭代后对当前最优解没有改进，则提前重之迭代
    - MAX_PATH_LENGTH: 每个整车路径允许的最大长度
    - COLONY_SIZE：每个迭代轮次中蚁群的大小（蚂蚁的个数）
    - MIN_PHEROMONE：信息素最小的浓度，为了避免过早收敛，当信息素矩阵中元素小于这个值时将被替换为该值
    - PLOT_INTERVAL：画图的轮次，10表示每10轮输出一次信息素和解的图像。注意画图为耗时操作，尤其当问题规模很大时画图将严重影响迭代速度