"""
Ant Colony optimizer
包含如下对象：
Ant: 蚂蚁，包含动作移动，结束，更新路径。
Colony：创建一个包含N个蚂蚁的Colony，可以设置初始信息素路径，迭代并记录每一个蚂蚁的路径。
"""

import random
from itertools import pairwise, permutations
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import threshold_accepting_tsp


class Ant(object):
    """
    普通蚂蚁，动作是可以移动到下一个节点，更新tabu, 输出路线。
    属性和参数:
        alpha: 信息启发式因子，决定了蚂蚁选择之前走过的路径的可能性
        beta:  期望启发式因子，决定蚂蚁预期的成本选择路径的可能，Beta越大约有可能选择局部最优
        rho: 信息挥发因子，（1-rho）为残留因子
        gamma: 随机因子，蚂蚁忽略启发函数，完全随机选择路径的可能性
        load: 蚂蚁当前的负载
        G: 一个DiGraph(), 存储了当前剩余的可选路径
        start_from: 出发点
        method: 如果是prob_pick 则咱找启发式因子的概率选择，如果是max_pick则选择启发式因子里最大的一个
    """
    def __init__(self, alpha, beta, gamma, G, id,
                 fix_cost_func:callable=None,
                 max_path_length=10,
                 pheromone=None) -> None:
        self.id = id or 'Unknown'
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.G = G
        self.fix_cost_func = fix_cost_func
        self.path = []
        self.tabu = []
        self.load = 0
        self.current_loc = None
        self.destination = None
        self.status = 'waiting'
        self.max_path_length = max_path_length
        self.max_retry = 100
        self.current_retry = 0
        self.pheromone = pheromone

    def __repr__(self) -> str:
        """
        格式为 Ant {id} | status
        """
        return f"Ant {self.id} | {self.status}"

    def start(self, node, destination=0):
        """
        从节点node处出发设置current_loc=node, 并将node上对应的demands加入数值加入到self.load
        :self.status 变为start
        :param node: 出发点的id
        """
        self.current_loc = node
        self.destination = destination
        self.load += self.G.nodes[node]['demand']
        self.path.append(node)
        self.status = 'start'

    def restart(self):
        """
        清空load, path, loads
        重新回到出发点, 清空path, 调用restart函数self.retry+1
        """
        self.load = 0
        self.current_loc = self.path[0] if self.path else None
        self.path = [self.current_loc]
        self.load = self.G.nodes[self.current_loc]['demand']
        self.current_retry += 1
        self.status = 'restarted'
    
    def choose_next(self, possible_nodes):
        """
        从给定的Nodes中选择下一个。选择的规则是：
        1）投掷筛子，从0~1之间取值，如果结果大于self.gamma 则随机选择，否则启发函数选择：
        2）如果按照启发函数选择，则先取得每个edges上的信息素浓度，计算下增加下一个节点的demands后当前的费用。以这两个作为算子得到下一个节点的权重。
        3）根据权重算得下个节点的概率，以对应的概率随机选择一个节点。作为return value
        """
        # 投掷筛子：
        eta = []
        tau = []
        for i in possible_nodes:
            new_loads = self.load + self.G.nodes[i]['demand']
            if self.G.nodes[i]['type'] != 'L':
                fix_cost = self.fix_cost_func(new_loads) or 0
            ltl_cost = self.G[self.current_loc][i]['ltl_cost'] * new_loads or 0
            ftl_cost = self.G[self.current_loc][i]['ftl_cost'] or 0
            new_cost = fix_cost + ltl_cost + ftl_cost
            eta.append(1/new_cost)
            if self.pheromone is None:
                tau.append(self.G[self.current_loc][i]['pheromone'] )
            else:
                tau.append(self.pheromone[self.current_loc][i])

        weights = (np.array(eta) ** self.beta) * (np.array(tau) ** self.alpha)
        probs = weights / np.sum(weights)
        if  np.random.random() <= self.gamma:
            next_node = int(random.choices(possible_nodes, weights=probs)[0])
        else:
            next_node = int(possible_nodes[np.argmax(probs)])
        return next_node


    def move_next(self):
        """
        从当前的位置选择下一个节点，如果没有节点可选，则调用restart，返回出发点。
        选择节点的规则：
        1）如果当前path的长度已经等于最大允许路径长度self.max_path_length, 只允许选择self.destination作为下一个节点
        2）当前位置的所有下游节点
        3）通向对应节点弧的capacity属性大于等于当前的self_load
        4）目标节点不在path中
        5）目标节点不在tabu中
        在满足以上条件的节点中随机选取一个节点，更新self.current_loc, 在self.load中增加当前节点的id
        """
        if len(self.path) == self.max_path_length:
            if self.destination and nx.has_path(self.G, self.current_loc, self.destination):
                next_node = self.destination
                print("Move to next node: ", next_node)
            else:
                # Update tabu,
                self.update_tabu()
                self.restart()
                return
                print("Restart !")
        else:
            possible_nodes = [n for n in self.G.successors(self.current_loc)
                              if self.G[self.current_loc][n]['capacity'] >= self.load
                              and n not in self.path
                              and n not in self.tabu]

            if not possible_nodes:
                self.update_tabu()
                self.restart()
                return

            next_node = self.choose_next(possible_nodes)

        self.current_loc = next_node
        self.load += self.G.nodes[next_node]['demand']
        self.path.append(next_node)


    def update_tabu(self):
        """
        将当前所在位置加入self.tabu
        """
        if self.current_loc not in self.tabu:
            self.tabu.append(self.current_loc)

    def end(self):
        """
        结束搜素，更新self.status为"arrived"
        """
        self.status = "arrived"

    def search_route(self):
        """
        调用从当前位置出发，如果当前位置为空则返回错误。
        1）检查当前位置是否为self.destination, 如果是，则结束循环
        2）调用move_next()移动至下一个节点
        3）如果self.retry大于最大允许次数则结束搜索
        """
        if not self.current_loc:
            raise ValueError("当前位置为空，无法开始搜索")

        while self.current_loc != self.destination:
            if self.current_retry > self.max_retry:
                self.status = "failed"
                raise ValueError("Ant ID:", self.id ," 搜索超过最大允许次数")
                break
            self.move_next()
            if self.current_loc == self.destination:
                self.end()
                break

    def show_attr(self):
        """
        依次打印Ant当前所有attribute的名称和值
        """
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")


class Colony(object):
    """
    包含n个蚂蚁的蚁群系统，主要功能是载入信息素路径，使得每一个蚂蚁都按照规则依次出发，
    在所有蚂蚁出发后根据更新规则更新信息素。
    """

    def __init__(self, id, k, alpha, beta, gamma, rho, G,
                 max_path_length=10, 
                 pheromone=None, 
                 fix_cost_func:callable=None,
                 ) -> None:
        self.id = id
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.G = G.copy()
        self.size = G.number_of_nodes()
        self.fix_cost_func = fix_cost_func
        self.max_path_length = max_path_length
        self.pheromone = np.ones((self.size, self.size)) if pheromone is None else pheromone
        self.solution = np.zeros((self.size, self.size))
        self.ants = []
        self.add_ants()

    def add_ants(self):
        """
        向蚁群中添加蚂蚁
        """
        self.ants = [Ant(id = self.id +"|"+ str(i),
                         alpha=self.alpha,
                         beta=self.beta,
                         gamma=self.gamma,
                         G=self.G,
                         max_path_length=self.max_path_length,
                         fix_cost_func=self.fix_cost_func,
                         pheromone=self.pheromone,
                         )
                     for i in range(self.k)]

    def iterate(self):
        """
        从蚂蚁队列中获得一只新的蚂蚁
        从剩余的所有节点上选择一个节点作为出发点，调用ant的start()方法
        调用蚂蚁的.search_route() 方法直到蚂蚁到达路径终点
        打印蚂蚁的最终路径，将该路径涉及的所有nodes（除源点外）从图上移除。
        如果路径中没有涉及type为"L"的节点，则调用Local search方法，按照distance重新排列节点。
        将path中的所有edge在solution矩阵中标记为1，其余为0
        """
        for ant in self.ants:
            f_nodes = [node for node in self.G.nodes() if self.G.nodes[node]['type'] == 'F']
            if f_nodes:
                #tart_node = max(f_nodes, key=lambda node: self.G.nodes[node]['demand'])
                start_node = int(np.random.choice(f_nodes))
            else:
                return "no feasible nodes"
            ant.start(start_node)
            ant.search_route()
            #print(f"蚂蚁的最终路径: {ant.path}")

            #if not any(self.G.nodes[node]['type'] == 'L' for node in ant.path):
                #ant.path = self.local_optimal(ant.path)

            for node in ant.path[:-1]:
                self.G.remove_node(node)
            for i in range(len(ant.path) - 1):
                self.solution[ant.path[i]][ant.path[i+1]] = 1

    def local_optimal(self, path):
        """
        使用path中的所有节点创建一个子图， 使用self.G中对应的nodes和edge 和 distance信息。
        调用threshold_accepting_tsp方法，从0点出发获得一个最优路径，return这个路径
        """
        subgraph = self.G.subgraph(path)
        for n in list(subgraph.nodes())[1:]:
            # Set path (0, n) to distance 0, to simulate an open end of a path
            try:
                subgraph[0][n]['distance'] = 0
            except KeyError:
                pass
        complete_g = nx.complete_graph(subgraph)
        optimal_path = threshold_accepting_tsp(complete_g, weight='distance',
                                               init_cycle=[0] + path, source=0)
        #print("Optimal path: ", optimal_path)
        return optimal_path[1:]

    def evaporate_and_yield(self):
        """
        按照比例，输出一个新的pheromone矩阵。
        self.pheromone *(1-rho） + rho * self.solution
        """
        return  (1-self.rho) * self.pheromone + self.rho * self.solution

    def collect_routs(self):
        """
        收集所有状态为arrived的ants 的paths和load, 返回一个dict
        """
        val = []
        for a in self.ants:
            if a.status == 'arrived':
                val.append({"weight": a.load, 
                            "path": a.path})
        return val


def generate_routes_from_pheromone(graph, pheromone):
    graph = graph.copy()
    # 将信息素添加到图中
    for i,j in graph.edges:
        graph[i][j]['weight'] = pheromone[i][j]
    solution = np.zeros_like(pheromone)
    # 取得所有类型为F的点  
    f_nodes = [node for node in graph if graph.nodes[node]['type'] == 'F'] + [0]
    #对于每个节点，只保留信息素浓度最大的下游节点
    for node in f_nodes:
        max_weight_successor = max(graph.successors(node), key=lambda n: graph[node][n]['weight'])
        edge_to_remove = []
        for successor in graph.successors(node):
            if successor != max_weight_successor:
                edge_to_remove.append((node, successor))
        graph.remove_edges_from(edge_to_remove)
    #取得所有类型为L的节点，如果入度为0表示没有需求被运送到这个节点，移除这类无入度的节点
    l_nodes = [node for node in graph if graph.nodes[node]['type'] == 'L']
    for node in l_nodes:
        if graph.in_degree(node) == 0:
            graph.remove_node(node)
    #取得所有入度为0的叶子节点，找到他们到0节点的最短路径（实际应该只有一个路径了）
    #计算这个路径的总需求返回
    leaves = [node for node in graph if graph.in_degree(node)==0]
    result = []
    for l in leaves:
        path = nx.shortest_path(graph, source=l, target=0)
        demands = sum([graph.nodes[x]['demand'] for x in path])
        result.append({"path": path, "weight": demands})
        for i,j in pairwise(path):
            solution[i][j] = 1
    return result, solution

def generate_solution_from_routes(oldsolution,route):
    new_solution = np.zeros_like(oldsolution)
    for r in route:
        for i,j in pairwise(r['path']):
            new_solution[i][j] = 1
    return new_solution

if __name__ == "__main__":
    from optimizer.problem_graph import graphProblem



    plm = graphProblem(data_path=r"D:\Development\code_commit_repo\vrp\dataset\test_data_100_nodes\data.pkl",
                       output_path=r"D:\Development\code_commit_repo\vrp\outputs",)
    plm.set_pheromones()
    plm.show_graph()


    print("测试随机选择节点的函数")
    starts = 10
    a = Ant(alpha=0.5, beta=0.5, gamma=1,
        id='test', G=plm.G, max_path_length=10,
        fix_cost_func=plm.ftl_fix_cost)
    a.start(starts)
    possible_nodes = [k for k in plm.G.successors(starts)]
    print("备选节点节点：", possible_nodes)
    print("选择节点：",a.choose_next(possible_nodes))
    print("自动探索")
    a.search_route()
    print("最终路径:", a.path)
    a.show_attr()

    print("测试蚁群")
    colony = Colony(id='Test_colony', k=100, alpha=0.5, beta=0.5, gamma=1,
                    rho=0.5, G=plm.G, 
                    max_path_length=6,
                    fix_cost_func=plm.ftl_fix_cost,
                    pheromone=None) 
    print("开始依次遍历")
    colony.iterate()
    print("打印每只蚂蚁的状态：")
    for a in colony.ants:
        print(a, 'route:', a.path, 'loads:', a.load)
    
    print("最终结果")
    print(colony.solution)

    plm.show_graph(highlights=colony.solution)
    
    print("计算费用")
    routes = colony.collect_routs()
    cost = plm.calculate_cost(routes)
    print("该方案费用为:", cost)

    print("更新信息素")
