"""
暴力生成所有的排列组合以便得知最优解的位置。
针对于非常小规模的用于测试的数据集
"""
import os

import numpy as np
import pandas as pd
from optimizer.problem_graph import graphProblem
from itertools import permutations, product, compress

#加载原问题用于计算成本
DATA_PATH = r'D:\Development\code_commit_repo\vrp\dataset\test_data_5_nodes\data.pkl'
plm = graphProblem(data_path=DATA_PATH, output_path="")


#将原问题视为一个5个节点的排列组合，5个节点可以最多用四个分割，从'的位置'为可分割的子集。
SIZE = 5
all_nodes = [i for i in range(1, SIZE+1)]+ ['x','x','x', 'x']
all_permutations = set(permutations(all_nodes))
print(f"共获得:{len(all_permutations)}个方案")

# 从X处将每个排列组合分割
split_solutions = []
for perm in all_permutations:
    temp = []
    subset = []
    for i in range(len(perm)):
        if perm[i] != 'x':
            subset.append(perm[i])
        elif subset:
            temp.append(subset)
            subset = []
    if subset:
        temp.append(subset)
    if temp:
        split_solutions.append(temp)

# 再次遍历每个solution， 对于其中有单个节点的成员，再衍生出一个带有ltl节点的
drived_solutions = []
for perm in split_solutions:
    long_routs = []
    single_ele = []
    for subset in perm:
        if len(subset) == 1:
            single_ele.append(subset)
        else:
            long_routs.append(subset)
    if not single_ele:
        drived_solutions.append(long_routs)
    else:
        bitmap_comb = product([0,1], repeat=len(single_ele))
        for b in bitmap_comb:
            ftl = [i for i in compress(single_ele,b)]
            ltl = [i + [i[0] + SIZE] for i in compress(single_ele,tuple(1 - x for x in b))]
            drived_solutions.append(long_routs + ftl + ltl)
print(f"共获得:{len(drived_solutions)}个方案")

#重新整理所有的solutions
final_costs = []
solution_weights = []
for s in drived_solutions:
    temp = []
    for r in s:
        path = r + [0]
        demands = sum(plm.G.nodes[n]['demand'] for n in path)
        temp.append({"path":path, 'weight': demands})
    try:
        final_costs.append(plm.calculate_cost(temp))
    except KeyError as e:
        print("不可行解", s)
        final_costs.append(np.inf)
    solution_weights.append(temp)

output_path = "\\".join(DATA_PATH.split("\\")[:-1])
data = pd.DataFrame({"Solutions":[str(s) for s in solution_weights], "Costs": final_costs})
data.drop_duplicates().to_excel(os.path.join(output_path,"brute_force_result.xlsx"))