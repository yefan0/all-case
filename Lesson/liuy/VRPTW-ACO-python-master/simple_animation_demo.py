"""
简单的动画演示脚本 - 生成VRPTW优化过程的静态图片序列
"""

import matplotlib.pyplot as plt
from vrptw_base import VrptwGraph
from basic_aco import BasicACO
import os


def create_simple_animation():
    """创建简单的动画图片序列"""
    print("="*60)
    print("生成VRPTW蚁群优化算法演示图片")
    print("="*60)
    
    # 创建保存目录
    save_path = "./simple_animation/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 使用c101数据集
    file_path = './solomon-100/c101.txt'
    
    # 参数设置
    ants_num = 8
    max_iter = 15
    beta = 2
    q0 = 0.1
    
    print(f"数据文件: {file_path}")
    print(f"蚂蚁数量: {ants_num}")
    print(f"最大迭代次数: {max_iter}")
    print(f"保存路径: {save_path}")
    
    # 创建图对象
    graph = VrptwGraph(file_path)
    
    # 存储最佳路径的历史
    best_paths_history = []
    best_distances_history = []
    
    # 自定义ACO类来收集历史数据
    class CustomACO(BasicACO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.paths_history = []
            self.distances_history = []
        
        def _basic_aco(self, path_queue_for_figure):
            import time
            start_time_total = time.time()
            start_iteration = 0
            
            for iter in range(self.max_iter):
                # 运行一次迭代
                ants = [self.create_ant() for _ in range(self.ants_num)]
                
                for k in range(self.ants_num):
                    while not ants[k].index_to_visit_empty():
                        next_index = self.select_next_index(ants[k])
                        if not ants[k].check_condition(next_index):
                            next_index = self.select_next_index(ants[k])
                            if not ants[k].check_condition(next_index):
                                next_index = 0
                        
                        ants[k].move_to_next_index(next_index)
                        self.graph.local_update_pheromone(ants[k].current_index, next_index)
                    
                    ants[k].move_to_next_index(0)
                    self.graph.local_update_pheromone(ants[k].current_index, 0)
                
                # 计算最佳路径
                import numpy as np
                paths_distance = np.array([ant.total_travel_distance for ant in ants])
                best_index = np.argmin(paths_distance)
                
                if self.best_path is None or paths_distance[best_index] < self.best_path_distance:
                    self.best_path = ants[int(best_index)].travel_path
                    self.best_path_distance = paths_distance[best_index]
                    self.best_vehicle_num = self.best_path.count(0) - 1
                    start_iteration = iter
                    
                    # 记录历史
                    self.paths_history.append(self.best_path[:])
                    self.distances_history.append(self.best_path_distance)
                    
                    print(f'[iteration {iter}]: find improved path, distance = {self.best_path_distance:.2f}')
                
                # 更新信息素
                self.graph.global_update_pheromone(self.best_path, self.best_path_distance)
                
                # 检查收敛
                if iter - start_iteration > 50:
                    print(f'iteration exit: no better solution in 50 iterations')
                    break
            
            print(f'final best path distance is {self.best_path_distance:.2f}, vehicles: {self.best_vehicle_num}')
            return self.paths_history, self.distances_history
        
        def create_ant(self):
            from ant import Ant
            return Ant(self.graph)
    
    # 运行算法
    print("\n开始运行算法...")
    custom_aco = CustomACO(graph, ants_num=ants_num, max_iter=max_iter, beta=beta, q0=q0,
                           whether_or_not_to_show_figure=False)
    
    paths_history, distances_history = custom_aco._basic_aco(None)
    
    # 生成图片序列
    print(f"\n生成 {len(paths_history)} 张优化过程图片...")
    
    for i, (path, distance) in enumerate(zip(paths_history, distances_history)):
        plt.figure(figsize=(12, 10))
        
        # 绘制节点
        plt.scatter([graph.nodes[0].x], [graph.nodes[0].y], 
                   c='red', label='Depot', s=100, marker='s')
        plt.scatter([node.x for node in graph.nodes[1:]], 
                   [node.y for node in graph.nodes[1:]], 
                   c='steelblue', label='Customers', s=50, marker='o')
        
        # 绘制路径
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        vehicle_paths = []
        current_path = []
        
        for j in range(len(path)):
            current_path.append(path[j])
            if j > 0 and path[j] == 0:
                vehicle_paths.append(current_path[:])
                current_path = [0]
        
        for vehicle_idx, vehicle_path in enumerate(vehicle_paths):
            if len(vehicle_path) > 1:
                color = colors[vehicle_idx % len(colors)]
                
                for k in range(1, len(vehicle_path)):
                    x_list = [graph.nodes[vehicle_path[k-1]].x, graph.nodes[vehicle_path[k]].x]
                    y_list = [graph.nodes[vehicle_path[k-1]].y, graph.nodes[vehicle_path[k]].y]
                    
                    plt.plot(x_list, y_list, color=color, linewidth=2, 
                            label=f'Vehicle {vehicle_idx+1}' if k == 1 else "", alpha=0.8)
        
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.title(f'VRPTW Solution - Iteration {i+1}\nDistance: {distance:.2f}, Vehicles: {path.count(0)-1}', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # 保存图片
        filename = f"frame_{i+1:03d}_dist_{distance:.1f}.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"保存: {filename}")
    
    # 创建最终结果图
    if custom_aco.best_path:
        plt.figure(figsize=(14, 12))
        
        # 绘制节点
        plt.scatter([graph.nodes[0].x], [graph.nodes[0].y], 
                   c='red', label='Depot', s=150, marker='s', zorder=5)
        plt.scatter([node.x for node in graph.nodes[1:]], 
                   [node.y for node in graph.nodes[1:]], 
                   c='steelblue', label='Customers', s=80, marker='o', zorder=4)
        
        # 添加节点标签
        for i, node in enumerate(graph.nodes):
            if i == 0:
                plt.annotate('Depot', (node.x, node.y), 
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=10, color='red', weight='bold')
            else:
                plt.annotate(f'{i}', (node.x, node.y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='blue')
        
        # 绘制最终路径
        path = custom_aco.best_path
        vehicle_paths = []
        current_path = []
        
        for i in range(len(path)):
            current_path.append(path[i])
            if i > 0 and path[i] == 0:
                vehicle_paths.append(current_path[:])
                current_path = [0]
        
        for vehicle_idx, vehicle_path in enumerate(vehicle_paths):
            if len(vehicle_path) > 1:
                color = colors[vehicle_idx % len(colors)]
                
                for i in range(1, len(vehicle_path)):
                    x_list = [graph.nodes[vehicle_path[i-1]].x, graph.nodes[vehicle_path[i]].x]
                    y_list = [graph.nodes[vehicle_path[i-1]].y, graph.nodes[vehicle_path[i]].y]
                    
                    plt.plot(x_list, y_list, color=color, linewidth=2.5, 
                            label=f'Vehicle {vehicle_idx+1}' if i == 1 else "", alpha=0.8, zorder=3)
        
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Y Coordinate', fontsize=14)
        plt.title(f'VRPTW Final Solution\nTotal Distance: {custom_aco.best_path_distance:.2f}, Vehicles: {custom_aco.best_vehicle_num}', 
                 fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=10)
        
        final_path = os.path.join(save_path, "final_best_solution.png")
        plt.savefig(final_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"保存最终解决方案: final_best_solution.png")
    
    print("\n" + "="*60)
    print("动画图片序列生成完成！")
    print(f"共生成 {len(paths_history)} 张优化过程图片")
    print(f"文件保存在: {save_path}")
    print("="*60)


if __name__ == '__main__':
    create_simple_animation() 