"""
演示脚本 - 生成完整的VRPTW优化过程动画（包括GIF）
"""

from vrptw_base import VrptwGraph
from basic_aco import BasicACO
from save_animation_figure_en import SaveAnimationFigureEN
from threading import Thread
from queue import Queue
from vrptw_base import PathMessage


def create_animation_gif():
    """创建VRPTW优化过程的GIF动画"""
    print("="*60)
    print("VRPTW蚁群优化算法动画演示")
    print("="*60)
    
    # 使用c101数据集
    file_path = './solomon-100/c101.txt'
    
    # 参数设置（适中的参数以获得较好的演示效果）
    ants_num = 8
    max_iter = 20
    beta = 2
    q0 = 0.1
    
    print(f"数据文件: {file_path}")
    print(f"蚂蚁数量: {ants_num}")
    print(f"最大迭代次数: {max_iter}")
    print(f"动画保存路径: ./final_animation/")
    print("\n开始运行算法并生成动画...")
    
    # 创建图对象
    graph = VrptwGraph(file_path)
    
    # 创建队列用于传递路径信息
    path_queue_for_figure = Queue()
    
    # 创建保存动画的可视化对象（英文版，避免字体问题）
    save_figure = SaveAnimationFigureEN(graph.nodes, path_queue_for_figure, 
                                        save_path="./final_animation/")
    
    # 定义算法运行函数
    def run_algorithm():
        print("算法开始运行...")
        basic_aco = BasicACO(graph, ants_num=ants_num, max_iter=max_iter, beta=beta, q0=q0,
                             whether_or_not_to_show_figure=False)
        basic_aco._basic_aco(path_queue_for_figure)
        
        # 保存最终结果的高质量图片
        if basic_aco.best_path:
            save_figure.save_final_result(basic_aco.best_path, basic_aco.best_path_distance, 
                                         basic_aco.best_vehicle_num, "best_solution.png")
        
        # 发送结束信号
        path_queue_for_figure.put(PathMessage(None, None))
        print("算法运行完成")
    
    # 在新线程中运行算法
    algorithm_thread = Thread(target=run_algorithm)
    algorithm_thread.start()
    
    # 运行并保存动画（保存图片帧和GIF）
    print("开始保存动画帧和生成GIF...")
    save_figure.run_and_save(save_static=True, save_gif=True)
    
    # 等待算法线程完成
    algorithm_thread.join()
    
    print("\n" + "="*60)
    print("动画生成完成！")
    print("生成的文件:")
    print("1. 静态图片帧: ./final_animation/frame_*.png")
    print("2. GIF动画: ./final_animation/vrptw_optimization_process.gif")
    print("3. 最终解决方案: ./final_animation/best_solution.png")
    print("="*60)


def create_comparison_images():
    """创建算法对比图"""
    print("\n生成算法对比图...")
    
    file_path = './solomon-100/c101.txt'
    graph = VrptwGraph(file_path)
    
    # 创建对比图保存目录
    import os
    if not os.path.exists('./comparison/'):
        os.makedirs('./comparison/')
    
    save_figure = SaveAnimationFigureEN(graph.nodes, None, save_path="./comparison/")
    
    # 运行基础ACO算法
    print("运行基础ACO算法...")
    basic_aco = BasicACO(graph, ants_num=10, max_iter=30, beta=2, q0=0.1, 
                         whether_or_not_to_show_figure=False)
    basic_aco.run_basic_aco()
    
    # 保存基础ACO结果
    if basic_aco.best_path:
        save_figure.save_final_result(basic_aco.best_path, basic_aco.best_path_distance, 
                                     basic_aco.best_vehicle_num, "basic_aco_solution.png")
        print(f"基础ACO结果: 距离={basic_aco.best_path_distance:.2f}, 车辆数={basic_aco.best_vehicle_num}")
    
    print("对比图生成完成！")


if __name__ == '__main__':
    # 创建动画GIF
    create_animation_gif()
    
    # 创建对比图
    create_comparison_images()
    
    print("\n所有演示文件生成完成！") 