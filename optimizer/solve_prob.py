"""
求解问题
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from openpyxl import Workbook
import time

from optimizer.problem_graph import graphProblem
from optimizer.aco import Colony, generate_routes_from_pheromone, generate_solution_from_routes
from optimizer.local_optimal import apply_local_optimal


def add_matrix_to_ws(ws, data):
    """
    将data中的数增添到ws现有数据区域下方，保持matrix的行列形状不变。
    :param ws: openpyxl的excel worksheet
    :param data: 一个numpy 二维数组或者矩阵
    """
    # 获取当前工作表的最大行数
    max_row = ws.max_row
    # 遍历data中的每一行
    for row_index, row in enumerate(data, start=1):
        # 遍历每一行中的每一个元素
        for col_index, value in enumerate(row, start=1):
            # 将元素写入到工作表中，位置是当前最大行数加上行索引
            ws.cell(row=max_row + row_index, column=col_index, value=value)
    

def solve_problem(data_path, output_path, alpha, beta, rho, gamma,
                  max_loop, no_improvement_threshold,
                  max_path_length, colony_size, min_pheromone, plot_interval):
    print("开始加载待求解问题")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, 'img'))
        os.makedirs(os.path.join(output_path, 'img', 'solution'))
        os.makedirs(os.path.join(output_path, 'img', 'pheromone'))

    plm = graphProblem(data_path=data_path, output_path=output_path)
    plm.set_pheromones()
    # plm.show_graph()
    nodes_size = len([f for f in plm.G.nodes if plm.G.nodes[f]['type'] == 'F'])

    best_so_far_solution = None
    best_so_far_cost = float('inf')
    no_improvement_count = 0
    pheromone = None
    his_costs = []
    his_solutions = []
    his_pheromones = []
    his_routes = []
    gamma_init = gamma
    start_time = time.time()  # 开始计时

    for i in tqdm(range(max_loop), desc="ACO Interation: ", leave=False):
        colony = Colony(id=f'Colony|{i}',
                        k=colony_size,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        rho=rho,
                        G=plm.G,
                        max_path_length=max_path_length,
                        fix_cost_func=plm.ftl_fix_cost,
                        pheromone=pheromone if pheromone is not None else plm.pheromone)
        colony.iterate()
        solution = colony.solution
        routes = colony.collect_routs()
        cost = plm.calculate_cost(solution_path=routes)
        his_costs.append(cost)
        his_pheromones.append(pheromone if pheromone is not None else plm.pheromone)
        his_solutions.append(solution)
        tqdm.write(f"找到路径: {len(routes)}条")
        if cost < best_so_far_cost:
            opt_routes, opt_cost = apply_local_optimal(solution=routes,
                                                       cost_func=plm.calculate_cost,
                                                       nodes_size=nodes_size)
            if opt_cost < cost:
                cost = opt_cost
                solution = generate_solution_from_routes(solution, opt_routes)
                his_costs[-1] = cost
                his_solutions[-1] = solution
            his_routes.append(opt_routes)
            pheromone =  (1 - rho)  * colony.pheromone + rho * solution
            best_so_far_solution = solution
            best_so_far_cost = cost
            no_improvement_count = 0
        else:
            his_routes.append(routes)
            no_improvement_count += 1
            rank_new_cost = np.argsort(his_costs)[-1]
            pheromone =  (1 - rho)  * pheromone + rho * (rank_new_cost/len(his_costs) * solution
                                                     +  best_so_far_solution)

        if no_improvement_count >= no_improvement_threshold:  # 如果大于NO_IMPRO_THRESHOLD的循环次数内没有改进，则停止
            break
        gamma = gamma * gamma
        pheromone = np.where(pheromone < min_pheromone, min_pheromone, pheromone)
        if (i % plot_interval == 0) and (plot_interval > 0):
            plm.show_graph(highlights=solution, filename=f'img/solution/{i}.png')
            plm.show_graph(highlights=pheromone * 10, filename=f'img/pheromone/{i}.png')
        tqdm.write(f"当前成本: {cost}")

    print("最终结果")
    total_iteration = i
    print("Total iteration:", total_iteration)
    print("Best Sofar Solution:", best_so_far_solution)
    print("Best Sofar Cost:", best_so_far_cost)
    plm.show_graph(highlights=best_so_far_solution*2, filename="best_sofar_solution.png")
    result,solution = generate_routes_from_pheromone(plm.G, pheromone)
    plm.show_graph(highlights=solution*2, filename="final_solution.png")
    final_cost = plm.calculate_cost(result)
    print('最终方案的成本为：', final_cost)

    # 将结果输出到Excel文件中

    wb = Workbook()
    ws_solution = wb.active
    ws_solution.title = 'Solution'

    ws_solution.append([str(r) for r in result])
    ws_pheromone = wb.create_sheet('Pheromone')
    ws_solution.append(["Costs"])
    ws_solution.append(his_costs)

    ws_routes = wb.create_sheet('Pheromone')
    ws_routes.title = 'Routes'
    ws_routes.append(['Costs','Routes'])


    for i, (p, s, rs, c) in enumerate(zip(his_pheromones, his_solutions, his_routes, his_costs)):
        ws_solution.append([f"Solution Iter {i}"])
        add_matrix_to_ws(ws_solution,s)
        ws_pheromone.append([f"Pheromone Iter {i}"])
        add_matrix_to_ws(ws_pheromone, p)
        ws_routes.append([c]+[str(r) for r in rs])
    wb.save(output_path + r'\result.xlsx')


    # Print best sofar cost
    plt.plot(his_costs)
    plt.scatter(his_costs.index(best_so_far_cost), best_so_far_cost, color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Over Iterations')
    plt.savefig(output_path + fr'\img\coi_a{str(alpha).replace(".","_")}_b{str(beta).replace(".","_")}_g{str(gamma_init).replace(".","_")}_r{str(rho).replace(".","_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    end_time = time.time()  # 结束计时
    time_eclipse = time.strftime("%H:%M:%S",
                                 time.gmtime(end_time - start_time)) + f".{int((end_time - start_time) * 1000) % 1000:03d}"
    print(f"迭代开始到终止所用的时间: {end_time - start_time}秒")

    return {'best_sofar_solution': best_so_far_solution,
            'best_sofar_cost': best_so_far_cost,
            'first_cost': his_costs[0],
            'final_cost': final_cost,
            'total_iteration': i,
            'time_eclipse': end_time - start_time,
            'his_costs': his_costs,}


if __name__ == "__main__":

    data_path = r'D:\Development\code_commit_repo\vrp\dataset\test_data_100_nodes\data.pkl'
    output_path = r'D:\Development\code_commit_repo\vrp\dataset\test_data_100_nodes\result'
    import gc
    from itertools import product
    import matplotlib

    #matplotlib.use('Agg') #使用AGG为不展示图片
    MAX_LOOP = 1000
    NO_IMPRO_THRESHOLD = 100
    MAX_PATH_LENGTH = 4
    COLONY_SIZE = 110
    MIN_PHEROMONE = 0.001
    PLOT_INTERVAL = 1000
    ATTEMPTS = 1

    alpha_list = [0.05,0.5,1]
    beta_list =[0.5,1,3]
    gamma_list = [0.99,0.95,1]
    rho_list = [0.01,0.05,0.1]

    import pandas as pd
    from tqdm import tqdm
    res_list = []
    for alpha, beta, rho, gamma in tqdm(list(product(alpha_list, beta_list, rho_list, gamma_list)), desc="Parameter Tuning"):
        for i in range(ATTEMPTS):
            try:
                res = solve_problem(data_path, output_path,
                                    alpha, beta, rho, gamma, MAX_LOOP,
                                    NO_IMPRO_THRESHOLD,
                                    MAX_PATH_LENGTH,
                                    COLONY_SIZE,
                                    MIN_PHEROMONE, PLOT_INTERVAL)
                gc.collect()


                res_list.append({**res, 'alpha': alpha, 'beta': beta, 'rho': rho, 'gamma': gamma,'attempt':i})
            except Exception as e:
                print(e)
    res_df = pd.DataFrame(res_list)
    res_df.to_excel(output_path + r'\param_tune.xlsx', index=False)