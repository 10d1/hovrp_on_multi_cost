"""
这个文件用于创建一个可以被蚁群访问的网络类，主要功能是基于networkx的DiGraph创建一个新的对象。
新的类包含如下属性和功能:
1) 记录每轮搜索中产生的最优路径，记录每轮的信息素。
2）除了网络的链接外，存储费用，路程，需求等属性
3）打印最优解，打印求解收敛的过程
"""

import networkx as nx
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from itertools import pairwise


class graphProblem(object):
    """
    包含DiGraph的问题
    初始参数：
    param data_path: 输入路径，指向了存储了图基本信息的pickle文件。
    param output_path: 输出路径，用于输出问题产生的结果文件和图。
    """
    def __init__(self, data_path, output_path:str) -> None:
        self.G = nx.DiGraph()
        self.history = []   # List，每个元素为一个Dict,存储 solution, pheromone等每轮迭代的附加信息
        self.best_sofar_result = None
        self.data_path = data_path
        self.output_path = output_path
        self.__load_graph()
        self.size = self.G.number_of_nodes()
        self.pheromone = np.random.random((self.size, self.size))

    def __load_graph(self):
        """
        从类的data_path中读取pickle文件并将nodes, edges, 添加入self.G的图中。
        将文件的其他属性添加入类本身的属性.
        读入的pickle文件应包括：{
        'ftl_cost': ftl_cost,
        'nodes': ext_nodes,
        'edges': ext_edges,
        'distances': ext_distance,
        'nodes_type': ext_nodes_type,
        'ltl_cost': ltl_cost,
        'demands': demands
        }
        """
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
            self.data = data

        self.G.add_nodes_from(range(len(data['nodes'])))
        self.G.add_edges_from(data['edges'])
        nx.set_node_attributes(self.G, dict(zip(range(len(data['nodes_type'])), data['nodes_type'])), 'type')
        nx.set_node_attributes(self.G, dict(zip(range(len(data['demands'])), data['demands'])), 'demand')
        nx.set_node_attributes(self.G, dict(zip(range(len(data['nodes'])), data['nodes'])), 'pos')

        for i, j in self.G.edges():
            self.G[i][j]['distance'] = data['distances'][i][j]
            self.G[i][j]['ltl_cost'] = data['ltl_cost'][i][j]
            self.G[i][j]['capacity'] = data['capacity'][i][j]
            self.G[i][j]['ftl_cost'] = data['distances'][i][j] * data['ftl_cost']['cost_per_km']

        self.ftl_cost = data['ftl_cost']

    def save_stack(self, filename='model_stack.pkl'):
        """
        将对象中所有属性的值保存到一个pickle文件中。
        :param filename: 保存的位置为self.output_path + filename
        """
        with open(os.path.join(self.output_path, filename), 'wb') as f:
            pickle.dump(self.__dict__, f)

    def show_graph(self, highlights=None, filename=None):
        """
        展示self.G的图，如果给出filename，则存储到指定filename, 否则只用plt.show展示。
        展示规则为 nodes中的第一个点用红色源点展示，其余用蓝色小号x。
        :param highlights: 一个np矩阵，其中的元素i,j表示i到j的弧。如果传入的highlights非空，则在上图的graph基础上再叠加一个层，
        加粗显示所有highlights矩阵中不为0的弧。弧的粗细由highlights的值决定。
        """
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self.G,'pos')
        nx.draw_networkx_nodes(self.G, pos, node_color=['red'] + ['blue']*(len(self.G.nodes)-1),
                               node_size=[150 if self.G.nodes[node]['type'] == 'F' else 100 for node in self.G.nodes])
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', arrows=True)
        labels = {node: str(node) for node in self.G.nodes if self.G.nodes[node]['type'] == 'F'}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=15, font_color='darkgrey',
                                horizontalalignment='center', verticalalignment='center_baseline')
        
        if highlights is not None:
            for i, j in self.G.edges():
                try:
                    if highlights[i][j] != 0:
                        node_type = self.G.nodes[i]['type']
                        nx.draw_networkx_edges(self.G, pos, edgelist=[(i, j)], 
                                               edge_color='r' if node_type=='F' else 'g', 
                                               width=highlights[i][j], arrows=False)
                except IndexError:
                    pass
        if filename:
            plt.savefig(os.path.join(self.output_path, filename))
        else:
            plt.show()
        plt.close()

    def set_pheromones(self, phe_matrix=None):
        """
        传入一个pheromones矩阵作为初始值，将对应的信息更新进所有的edge的属性上。
        """
        if not phe_matrix:
            phe_matrix = np.random.random((self.size, self.size))
        self.pheromone = phe_matrix
        for i, j in self.G.edges():
            self.G[i][j]['pheromone'] = phe_matrix[i][j]

    def ftl_fix_cost(self, weight):
        for i, load in enumerate(self.ftl_cost['loads']):
            if weight <= load:
                return self.ftl_cost['fix_costs'][i]

    def calculate_cost(self, solution_path=None):
        """
        计算当前的cost, 分为三个部分, 每个路径的LTL成本, solution整体的FTL 里程成本和车辆固定成本。
        :param solution_path: list, 每个成员是一个dict{"weight":, "path": 表示路径的list[1，2，3]}
        """
        try:
            cost = 0
            for p in solution_path:
                cost += self.ftl_fix_cost(p['weight'])
                for i, j in pairwise(p['path']):
                    cost += self.G[i][j]['ltl_cost'] * p['weight']   # LTL成本
                    cost += self.ftl_cost['cost_per_km'] * self.G[i][j]['distance'] # FTL成本

            return cost
        except KeyError:
            #如果其中有edge不存在，直接返回无穷大
            return np.inf

if __name__=='__main__':
    plm = graphProblem(data_path=r"D:\Development\code_commit_repo\vrp\dataset\test_data_10_nodes\data.pkl",
                       output_path=r"D:\Development\code_commit_repo\vrp\outputs",)
    import numpy as np
    plm.show_graph(highlights=np.random.random((20,20))*5)
    total_cost = plm.calculate_cost([{'weight':1, 'path':[3,4,0]},{'weight':2, 'path':[10,7,0]}])
    print(total_cost)