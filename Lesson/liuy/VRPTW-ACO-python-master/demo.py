"""
VRPTW-ACO项目演示脚本
这个脚本展示了三种不同的算法使用方式，并支持保存动画
"""

import sys
import os
from vrptw_base import VrptwGraph
from basic_aco import BasicACO
from multiple_ant_colony_system import MultipleAntColonySystem
from save_animation_figure import SaveAnimationFigure


def demo_basic_aco():
    """演示基础蚁群算法"""
    print("="*60)
    print("演示 1: 基础蚁群算法 (Basic ACO)")
    print("="*60)
    
    # 参数设置
    file_path = './solomon-100/c101.txt'
    ants_num = 10
    max_iter = 50  # 减少迭代次数以便快速演示
    beta = 2
    q0 = 0.1
    show_figure = True
    
    print(f"数据文件: {file_path}")
    print(f"蚂蚁数量: {ants_num}")
    print(f"最大迭代次数: {max_iter}")
    print(f"启发信息重要性(beta): {beta}")
    print(f"贪婪选择概率(q0): {q0}")
    print(f"显示图形: {show_figure}")
    print("\n开始运行基础ACO算法...")
    
    graph = VrptwGraph(file_path)
    basic_aco = BasicACO(graph, ants_num=ants_num, max_iter=max_iter, beta=beta, q0=q0,
                         whether_or_not_to_show_figure=show_figure)
    
    basic_aco.run_basic_aco()
    
    print("\n基础ACO算法演示完成！")
    input("\n按回车键继续下一个演示...")


def demo_multiple_ant_colony_system():
    """演示多蚁群系统算法"""
    print("="*60)
    print("演示 2: 多蚁群系统算法 (MACS)")
    print("="*60)
    
    # 参数设置
    file_path = './solomon-100/c101.txt'
    ants_num = 10
    beta = 2
    q0 = 0.1
    show_figure = True
    
    print(f"数据文件: {file_path}")
    print(f"蚂蚁数量: {ants_num}")
    print(f"启发信息重要性(beta): {beta}")
    print(f"贪婪选择概率(q0): {q0}")
    print(f"显示图形: {show_figure}")
    print("\n开始运行多蚁群系统算法...")
    print("注意：此算法会同时优化旅行距离和车辆数量")
    
    graph = VrptwGraph(file_path)
    macs = MultipleAntColonySystem(graph, ants_num=ants_num, beta=beta, q0=q0, 
                                   whether_or_not_to_show_figure=show_figure)
    
    macs.run_multiple_ant_colony_system()
    
    print("\n多蚁群系统算法演示完成！")
    input("\n按回车键继续下一个演示...")


def demo_save_animation_basic_aco():
    """演示基础蚁群算法并保存动画"""
    print("="*60)
    print("演示 4: 基础蚁群算法 + 保存动画")
    print("="*60)
    
    # 参数设置
    file_path = './solomon-100/c101.txt'
    ants_num = 10
    max_iter = 30  # 减少迭代次数以便快速演示
    beta = 2
    q0 = 0.1
    
    print(f"数据文件: {file_path}")
    print(f"蚂蚁数量: {ants_num}")
    print(f"最大迭代次数: {max_iter}")
    print(f"启发信息重要性(beta): {beta}")
    print(f"贪婪选择概率(q0): {q0}")
    print(f"保存路径: ./animations/basic_aco/")
    print("\n开始运行基础ACO算法并保存动画...")
    
    graph = VrptwGraph(file_path)
    
    # 创建保存动画的ACO算法实例
    from threading import Thread
    from queue import Queue
    
    path_queue_for_figure = Queue()
    
    # 创建保存动画的可视化对象
    save_figure = SaveAnimationFigure(graph.nodes, path_queue_for_figure, 
                                      save_path="./animations/basic_aco/")
    
    # 在新线程中运行算法
    def run_basic_aco_thread():
        basic_aco = BasicACO(graph, ants_num=ants_num, max_iter=max_iter, beta=beta, q0=q0,
                             whether_or_not_to_show_figure=False)  # 关闭默认显示
        basic_aco._basic_aco(path_queue_for_figure)
        # 发送结束信号
        from vrptw_base import PathMessage
        path_queue_for_figure.put(PathMessage(None, None))
    
    # 启动算法线程
    algorithm_thread = Thread(target=run_basic_aco_thread)
    algorithm_thread.start()
    
    # 运行并保存动画
    save_figure.run_and_save(save_static=True, save_gif=True)
    
    # 等待算法线程完成
    algorithm_thread.join()
    
    print("\n基础ACO算法动画保存完成！")
    input("\n按回车键继续...")


def demo_save_animation_macs():
    """演示多蚁群系统算法并保存动画"""
    print("="*60)
    print("演示 5: 多蚁群系统算法 + 保存动画")
    print("="*60)
    
    # 参数设置
    file_path = './solomon-100/c101.txt'
    ants_num = 10
    beta = 2
    q0 = 0.1
    
    print(f"数据文件: {file_path}")
    print(f"蚂蚁数量: {ants_num}")
    print(f"启发信息重要性(beta): {beta}")
    print(f"贪婪选择概率(q0): {q0}")
    print(f"保存路径: ./animations/macs/")
    print("\n开始运行多蚁群系统算法并保存动画...")
    print("注意：此算法会同时优化旅行距离和车辆数量")
    
    graph = VrptwGraph(file_path)
    
    # 创建保存动画的MACS算法实例
    from multiprocessing import Process, Queue as MPQueue
    
    path_queue_for_figure = MPQueue()
    
    # 创建保存动画的可视化对象
    save_figure = SaveAnimationFigure(graph.nodes, path_queue_for_figure, 
                                      save_path="./animations/macs/")
    
    # 在新进程中运行算法
    def run_macs_process():
        macs = MultipleAntColonySystem(graph, ants_num=ants_num, beta=beta, q0=q0, 
                                       whether_or_not_to_show_figure=False)
        macs._multiple_ant_colony_system(path_queue_for_figure)
    
    # 启动算法进程
    algorithm_process = Process(target=run_macs_process)
    algorithm_process.start()
    
    # 运行并保存动画
    save_figure.run_and_save(save_static=True, save_gif=True)
    
    # 等待算法进程完成
    algorithm_process.join()
    
    print("\n多蚁群系统算法动画保存完成！")
    input("\n按回车键继续...")


def demo_batch_test():
    """演示批量测试（不显示图形）"""
    print("="*60)
    print("演示 3: 批量测试模式")
    print("="*60)
    
    # 参数设置
    ants_num = 10
    beta = 1
    q0 = 0.1
    show_figure = False
    
    print(f"蚂蚁数量: {ants_num}")
    print(f"启发信息重要性(beta): {beta}")
    print(f"贪婪选择概率(q0): {q0}")
    print(f"显示图形: {show_figure}")
    print("\n开始批量测试几个数据集...")
    
    # 创建结果目录
    if not os.path.exists('./result'):
        os.makedirs('./result')
    
    # 测试几个不同的数据集
    test_files = ['c101.txt', 'c102.txt', 'r101.txt']
    
    for file_name in test_files:
        file_path = os.path.join('./solomon-100', file_name)
        if os.path.exists(file_path):
            print(f"\n正在处理: {file_name}")
            print("-" * 40)
            
            file_to_write_path = os.path.join('./result', file_name.split('.')[0] + '-result.txt')
            graph = VrptwGraph(file_path)
            macs = MultipleAntColonySystem(graph, ants_num=ants_num, beta=beta, q0=q0, 
                                           whether_or_not_to_show_figure=show_figure)
            macs.run_multiple_ant_colony_system(file_to_write_path)
        else:
            print(f"文件 {file_name} 不存在，跳过...")
    
    print("\n批量测试完成！")
    print("结果已保存到 ./result/ 目录中")


def create_static_comparison():
    """创建静态对比图"""
    print("="*60)
    print("演示 6: 创建静态对比图")
    print("="*60)
    
    file_path = './solomon-100/c101.txt'
    print(f"数据文件: {file_path}")
    print("正在生成静态对比图...")
    
    graph = VrptwGraph(file_path)
    
    # 创建输出目录
    if not os.path.exists('./static_results'):
        os.makedirs('./static_results')
    
    # 运行基础ACO算法
    print("运行基础ACO算法...")
    basic_aco = BasicACO(graph, ants_num=10, max_iter=30, beta=2, q0=0.1, 
                         whether_or_not_to_show_figure=False)
    basic_aco.run_basic_aco()
    
    # 保存基础ACO结果
    if basic_aco.best_path:
        save_figure = SaveAnimationFigure(graph.nodes, None, save_path="./static_results/")
        save_figure.save_final_result(basic_aco.best_path, basic_aco.best_path_distance, 
                                     basic_aco.best_vehicle_num, "basic_aco_result.png")
    
    # 运行MACS算法
    print("运行多蚁群系统算法...")
    macs = MultipleAntColonySystem(graph, ants_num=10, beta=2, q0=0.1, 
                                   whether_or_not_to_show_figure=False)
    macs.run_multiple_ant_colony_system()
    
    # 保存MACS结果
    if macs.best_path:
        save_figure.save_final_result(macs.best_path, macs.best_path_distance, 
                                     macs.best_vehicle_num, "macs_result.png")
    
    print("静态对比图生成完成！")
    print("文件保存在 ./static_results/ 目录中")
    input("\n按回车键继续...")


def main():
    """主函数 - 演示菜单"""
    print("欢迎使用 VRPTW-ACO 项目演示系统！")
    print("这是一个用蚁群优化算法解决带时间窗口车辆路径问题的Python实现")
    print("\n项目特点:")
    print("- 支持车辆容量约束")
    print("- 支持时间窗口约束")
    print("- 提供实时可视化")
    print("- 包含多种算法变体")
    print("- 支持动画保存功能")
    
    while True:
        print("\n" + "="*60)
        print("请选择要演示的算法:")
        print("1. 基础蚁群算法 (Basic ACO)")
        print("2. 多蚁群系统算法 (MACS)")
        print("3. 批量测试模式")
        print("4. 基础蚁群算法 + 保存动画")
        print("5. 多蚁群系统算法 + 保存动画")
        print("6. 创建静态对比图")
        print("7. 退出")
        print("="*60)
        
        choice = input("\n请输入您的选择 (1-7): ").strip()
        
        if choice == '1':
            try:
                demo_basic_aco()
            except Exception as e:
                print(f"运行基础ACO时出错: {e}")
                input("按回车键继续...")
        
        elif choice == '2':
            try:
                demo_multiple_ant_colony_system()
            except Exception as e:
                print(f"运行MACS时出错: {e}")
                input("按回车键继续...")
        
        elif choice == '3':
            try:
                demo_batch_test()
            except Exception as e:
                print(f"批量测试时出错: {e}")
                input("按回车键继续...")
        
        elif choice == '4':
            try:
                demo_save_animation_basic_aco()
            except Exception as e:
                print(f"运行基础ACO动画保存时出错: {e}")
                input("按回车键继续...")
        
        elif choice == '5':
            try:
                demo_save_animation_macs()
            except Exception as e:
                print(f"运行MACS动画保存时出错: {e}")
                input("按回车键继续...")
        
        elif choice == '6':
            try:
                create_static_comparison()
            except Exception as e:
                print(f"创建静态对比图时出错: {e}")
                input("按回车键继续...")
        
        elif choice == '7':
            print("感谢使用 VRPTW-ACO 演示系统！")
            break
        
        else:
            print("无效的选择，请重新输入")


if __name__ == '__main__':
    main() 