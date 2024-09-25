"""
本文件用于生成一个含有n个结点的图，使用图中心为作为源source，并连接所有云允许距离范围内的点，创建距离矩阵。
需要输入参数如下：
    SIZE：tuple,（x, y）分别表示图的宽度和高度（像素）
    RESOLUTION：分辨率：多少个个像素代表1KM
    N_NODES: 图的节点个数
    MAX_DISTANCE: 最大允许的距离，当两个点的距离超过这一阈值，不连接两个节点，与源点之间的链接不受此限制。
    DEMANDS_RANGE:tuple （min_demands, max_demands）对于每个节点，最大最小的需求允许范围
    DISTANCE_JITTER:tuple 距离抖动的范围，在直线距离的基础上增加随机增加一定范围的距离抖动，0为不抖动
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os


def gen_locations(size: tuple=(1000, 1000), n_nodes: int=10, plotting: bool=False, pic_path:str=None):
    """
    在给定的画布上产生n个节点，返回横坐标x和纵坐标y。
    :param size: tuple, 画布大小
    :param n_nodes: 除源点外包含多少个节点
    :param plotting: True则使用pyplot图，否则不画图。
    :return: 含有n+1个节点的list，形式为[(x0, y0),(x1, y1),(x2, y2)...],其中0节点为图的中心点。
    """
    import random
    
    width, height = size
    nodes = [(width//2, height//2)]  # 中心点作为第一个节点
    
    for _ in range(n_nodes):
        x = random.randint(0, width)
        y = random.randint(0, height)
        nodes.append((x, y))
    
    if plotting:
        x, y = zip(*nodes)
        plt.scatter(x[1:], y[1:], color='blue', marker='x', s=50)  # 其余节点用蓝色小号x显示
        plt.scatter(x[0], y[0], color='red', s=100)  # 第一个节点用红色大点显示
        if pic_path:
            plt.savefig(pic_path)
        plt.show()
    
    return nodes


def gen_edges(nodes: list, resolution: float = 1, max_distance: float = 200, distance_jitter: tuple = (0, 0), 
              plotting: bool = False, pic_path:str=None):
    """
    为给定的节点生成边。
    :param nodes: 输入节点的list，默认第一个节点为中心。
    :param resolution: 用于点坐标距离和真实距离(KM)的换算，如点a到b之间，使用两点坐标计算直线距离后乘以resolution，得到实际距离基数。
    :param max_distance: 两点之间允许的最大距离，当超过这个阈值后距离设为无穷。源节点（list中的第一个）和其他节点之间的边不受此限制。
    :param distance_jitter: 元组，包含两个0~1之间的小数，为模拟真实情况将直线距离稍作抖动，计算直线距离后在distance_jitter范围内生成一随机小数，最终两点实际距离为distance*(1+jitter)。
    :param plotting: True则使用pyplot图，否则不画图。
    :return distances: 一个numpy矩阵，矩阵元素Aij代表由上述逻辑生成的点i到点j之间的距离。
    """

    n = len(nodes)
    distances = np.zeros((n, n))
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            distance = math.sqrt((x1-x2)**2 + (y1-y2)**2) * resolution
            
            if distance_jitter != (0, 0):
                jitter = random.uniform(*distance_jitter)
                distance *= (1 + jitter)
            
            if i != 0 and j != 0 and distance > max_distance:
                distance = float('inf')
            else:
                edges.append((j, i))
                edges.append((i, j))
            distances[i][j] = distances[j][i] = distance

    if plotting:
        plt.figure(figsize=(10, 10))
        x, y = zip(*nodes)
       
        for i in range(n):
            for j in range(i+1, n):
                if distances[i][j] != float('inf'):
                    plt.plot([x[i], x[j]], [y[i], y[j]], color='lightblue', linewidth=0.5)
        plt.scatter(x[1:], y[1:], color='blue', marker='x', s=50)  # 其余节点用蓝色小号x显示
        plt.scatter(x[0], y[0], color='red', s=100)  # 第一个节点用红色大点显示
        if pic_path:
            plt.savefig(pic_path)
        plt.show()
    return distances, edges


def dummy_ltl_cost_function(loads, fix_cost, cost_per_km, distance, plotting=False, pic_path:str=None):
    """
    仿真的费用函数，用于生成模拟的零担和整车费用，为了模拟零担费用，在某个重量后超越整车费用，采用如下策略：
     1）生成整车费用的函数后，随机在最大载荷和1/2最大载荷之间取一点x
     2）该点的total_ftl_cost为该点载荷所在区间对应的最大fix_cost + distance * ftl_unit_cost
     3）用这个点对应的整车费用除以随机选取的载荷后再乘以1至2之间的一个系数，以这个费用作为该距离零担的单位费用
    :param loads: tuple， 由小至大每个车型对应的最大载荷
    :param fix_cost: tuple, 由小至大的每个车型对应的起步费用：需要与loads的tuple等长
    :param cost_per_km: float， FTL每公里的平均运输费用（所有车型统一）
    :param distance: float，需要计算的运输距离
    :param plotting: bool，True则使用pyplot画图，分别展示fixcost和ltlcost，否则不画图。x轴为0至最大载荷元组的最大值，y轴为总费用。
                     在图中用红色线展示所有区间的total_ftl_cost这个分段函数
                     用蓝色线展示每个载荷对应的ltl_cost直线
    :param pic_path: str, 存储图片的路径，如果为空，则只展示图片不存储，如果传入一个路径，则将图片存到指定路径
    :return: float，返回计算出的ltl费用
    """
    # 确保loads和fix_cost长度相同
    assert len(loads) == len(fix_cost), "loads和fix_cost的长度必须相同"
    # 生成整车费用函数
    def ftl_cost(weight):
        for i, load in enumerate(loads):
            if weight <= load:
                return fix_cost[i] + distance * cost_per_km
        return fix_cost[-1] + distance * cost_per_km

    # 随机选择一个点x
    max_load = loads[-1]
    x = random.uniform(max_load / 2, max_load)

    # 计算该点的total_ftl_cost
    total_ftl_cost = ftl_cost(x)

    # 计算零担单位费用
    ltl_unit_cost = (total_ftl_cost / x ) * random.uniform(1, 2)

    if plotting:
        weights = np.linspace(0, max_load, 1000)
        ftl_costs = [ftl_cost(w) for w in weights]
        ltl_costs = [ltl_unit_cost * w for w in weights]

        plt.figure(figsize=(10, 6))
        plt.plot(weights, ftl_costs, 'r-', label='FTL Cost')
        plt.plot(weights, ltl_costs, 'b-', label='LTL Cost')
        plt.xlabel('Weight (Ton)')
        plt.ylabel('Cost')
        plt.title('FTL vs LTL Cost')
        plt.legend()

        if pic_path:
            plt.savefig(pic_path)
        plt.show()

    return ltl_unit_cost


def extent_cost_graph(nodes, demands, edges, distances, ftl_cost, plotting_path: str= None):
    """
    扩展现有节点，为除了源节点以外的每个nodes扩展一个虚拟节点，并在这个基础上扩展distances矩阵，在原有矩阵的基础上创建两种新的连接。
        1- 虚拟ltl节点和源节点（nodes 中第一个成员）
        2- 虚拟节点和其对应的源节点。
    :param nodes: list, 包含每个节点横纵坐标的列表，默认第一个节点为源节点（source）
    :param demands: list, 每个节点的需求量
    :param edges: list(tuple, tuple): 包含所有边的组合
    :param distances: numpy矩阵，元素Aij表示节点i到j的距离
    :param ftl_cost: 一个dict，内容如下：
                    {'loads': tuple 由小至大每个车型对应的最大载荷,
                     'fix_costs': tuple 由小至大的每个车型对应的起步费用：需要与loads的tuple等长, 
                     'cost_per_km': float, 每公里的车辆形式费用，所有车型统一}
    :param plotting_path: 图片保存的路径, 如果传入一个路径形状的字符串，则对于每一类型为F的节点，保存一张由调用dummy_ltl_cost_function产生的图，命名规则为plotting_path/节点序号.png
    :return ext_nodes: 扩展的nodes列表，如果源节点默认为list第一个成员，ext_nodes应包含(n-1)*2个元素
    :return ext_edges: 扩展的edges列表，原列表相比，增加了虚拟节点到对应节点，虚拟节点到源节点的新边
    :return ext_nodes_type: list 与ext_nodes等长，以字母表示每个节点的类型，源为S, 传入的其余原始节点为F, 扩展的零担节点为L
    :return ext_distance: list，扩展的distance矩阵，除原矩阵的元素之外其余新元素赋值为0
    :return ltl_cost: 扩展的ltl费用矩阵，与ext_distance的形状一致，其中元素规则为：
           1）如果边连接的两个点的类型分别为F和L，则调用dummy_ltl_cost_function产生一个ltl费用，并为该弧赋值。
           2）如果边连接的两个点的类型为由L到F则费用为正无穷。
           3）其余的边费用为 0
    """
    n = len(nodes)
    ext_nodes = nodes + nodes[1:]  # 为每个非源节点添加一个虚拟节点
    ext_demands = demands + [0] * (n - 1)
    ext_nodes_type = ['S'] + ['F'] * (n - 1) + ['L'] * (n - 1)
    ext_edges = edges.copy()
    ext_distance = np.zeros((2*n-1, 2*n-1))
    ext_capacity = np.ones((2*n-1, 2*n-1)) * np.inf
    ext_distance[:n, :n] = distances
    max_capacity = max(ftl_cost.get('loads'))
    ltl_cost = np.zeros((2*n-1, 2*n-1))

    for i in range(n):
        for j in range(n):
            # 所有FTL有关的弧上限限制为车辆的最大负载
            ext_capacity[i][j] = max_capacity

    for i in range(1, n):
        # 虚拟ltl节点和源节点的连接
        ext_distance[0][n+i-1] = ext_distance[n+i-1][0] = distances[0][i]
        ext_edges.append((n+i-1, 0))
        # 虚拟节点和其对应的源节点的连接
        ext_distance[i][n+i-1] = ext_distance[n+i-1][i] = 0
        # 虚拟节点到对应需求节点之间的capacity等于该节点的需求量
        ext_capacity[i][n+i-1] = ext_demands[i]
        ext_edges.append((i, n+i-1))

        # 计算LTL费用
        ltl_unit_cost = dummy_ltl_cost_function(
            loads=ftl_cost['loads'],
            fix_cost=ftl_cost['fix_costs'],
            cost_per_km=ftl_cost['cost_per_km'],
            distance=distances[0][i],
            plotting=(plotting_path is not None),
            pic_path=f"{plotting_path}/node_{i}.png" if plotting_path else None
        )
        
        # 设置LTL费用
        ltl_cost[i][n+i-1] = ltl_unit_cost
        ltl_cost[n+i-1][i] = float('inf')  # 从L到F的费用设为无穷大
    
    return ext_nodes, ext_demands, ext_edges, ext_nodes_type, ext_distance, ltl_cost, ext_capacity


def gen_demands(nodes:list, demand_range:tuple):
    """
    为除了源节点以外的所有节点生成一个Demands，在Demands Range内随机选择一个数值。
    :param nodes: 所有节点的列表
    :param demand_range: 一个元组，包含需求范围的最小值和最大值
    :return demands：一个包含随机生成的数值的列表
    """
    demands = [0]
    for _ in range(len(nodes) - 1):
        demand = random.uniform(demand_range[0], demand_range[1])
        demands.append(round(demand, 2))
    return demands



if __name__=="__main__":
    import pickle
    import os

    ftl_cost = {"loads": (2, 10, 20),
                "fix_costs": (200, 500, 1500),
                "cost_per_km": 0.5}
    data_path = r"D:\Development\code_commit_repo\vrp\dataset\test_data_100_nodes\\"
    demands_range = (0.1, 15)
    n_nodes = 100
    resolution = 1
    nodes = gen_locations(n_nodes=n_nodes, plotting=False)
    distances, edges = gen_edges(nodes, resolution=resolution, max_distance=500, 
                                     plotting=True, pic_path=os.path.join(data_path,'graph.png'))
    print(distances)
    print("测试ltl函数！")
    ltl = dummy_ltl_cost_function(loads=ftl_cost['loads'],
                                  fix_cost=ftl_cost['fix_costs'],
                                  cost_per_km=ftl_cost['cost_per_km'],
                                  distance=100,
                                  plotting=True, 
                                  )
    print("LTL成本为:", ltl)
    demands = gen_demands(nodes, demands_range)

    (ext_nodes, ext_demands, 
     ext_edges, ext_nodes_type, 
     ext_distance, 
     ltl_cost, ext_capacity) = extent_cost_graph(nodes,
                                                 demands,
                                                 edges,
                                                 distances,
                                                 ftl_cost,
                                                 plotting_path=os.path.join(data_path,'cost_funcs'))


    # 保存数据到pickle文件
    data_to_save = {
        'ftl_cost': ftl_cost,
        'nodes': ext_nodes,
        'edges': ext_edges,
        'distances': ext_distance,
        'nodes_type': ext_nodes_type,
        'ltl_cost': ltl_cost,
        'demands': ext_demands,
        'capacity': ext_capacity,
    }
    
    with open(os.path.join(data_path, 'data.pkl'), 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print("数据已保存到", os.path.join(data_path, 'data.pkl'))