"""
对一个solution应用的局部优化算法
solution的结构[{"path":[1,2,3,0],"weight":9}] 每个dict包含solution中的一个path
"""
import numpy as np

def two_opt_swap(path, i, j):
    """
    n pseudocode, the mechanism by which the 2-opt swap manipulates a given route is as follows. Here v1 and v2 are the
     first vertices of the edges that are to be swapped when traversing through the route:
    function 2optSwap(route, v1, v2) {
    1. take route[start] to route[v1] and add them in order to new_route
    2. take route[v1+1] to route[v2] and add them in reverse order to new_route
    3. take route[v2+1] to route[start] and add them in order to new_route
    return new_route;
}
    """
    path = path.copy()
    i += 1
    while i < j:
        path[i], path[j] = path[j], path[i]
        i += 1
        j -= 1
    return path

def two_opt(path, weight, cost_func, iteration_num=50):
    """
    用于实现对于一个解决方案的局部优化，solution中的每一个path,使用2optSwap方法，然后调用
    cost_func计算cost, 如果cost有减少，则使用新的path代替原来的。
    """
    best_distance = cost_func([{"path":path, "weight": weight}])
    for _ in range(iteration_num):
        improved = False
        for i in range(len(path) - 2):
            for j in range(i + 1, len(path)-1):
                new_path = two_opt_swap(path[:-1], i, j) + [0]
                try:
                    new_distance = cost_func([{"path":new_path,"weight": weight}])
                except KeyError: #如果出现不可行的解
                    new_distance = np.inf
                if new_distance < best_distance:
                    path = new_path
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return path, best_distance


def insert_ltl(path, size, weight, cost_func):
    """
    去掉ltl节点看是否对于方案有提升
    """
    cost = cost_func([{"path":path, "weight": weight}])
    start = path[0]
    ltl = cost_func([{"path":[start, start + size, 0], "weight": weight}])
    if ltl <= cost:
        return [start, start + size, 0], ltl
    else:
        return path, cost


def apply_local_optimal(solution, nodes_size, cost_func, iteration_num=50):
    """
    对整个方案的所有path应用2opt
    """
    new_solution = []
    for r in solution:
        if len(r['path']) <= 2: #如果节点短，则尝试插入ltl看是否有改进
            new_path, _ = insert_ltl(path=r['path'],
                                     weight=r['weight'],
                                     cost_func=cost_func,
                                     size=nodes_size)
            new_solution.append({'path': new_path, 'weight': r['weight']})
        else:
            new_path, _ = two_opt(path=r['path'], 
                                  weight=r['weight'],
                                  cost_func=cost_func, 
                                  iteration_num=iteration_num)
            new_solution.append({'path': new_path, 'weight': r['weight']})
    return new_solution, cost_func(new_solution)


if __name__=="__main__":
    from optimizer.problem_graph import graphProblem
    print(two_opt_swap(['A','B','E','D','C','F','G','H','A'],1,4))

    solution = [{"path":[2,3,0],"weight":9},{"path":[1,5,4,0],"weight":9}]
    plm = graphProblem(data_path=r'D:\Development\code_commit_repo\vrp\dataset\test_data_5_nodes\data.pkl',
                       output_path=None)
    print(apply_2opt(solution, plm.calculate_cost))