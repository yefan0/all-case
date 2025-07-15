"""
测试保存动画功能的简单脚本
"""

from vrptw_base import VrptwGraph
from basic_aco import BasicACO
from save_animation_figure import SaveAnimationFigure
from threading import Thread
from queue import Queue
from vrptw_base import PathMessage


def test_save_animation():
    """测试保存动画功能"""
    print("开始测试保存动画功能...")
    
    # 使用小规模数据集进行快速测试
    file_path = './solomon-100/c101.txt'
    
    # 参数设置（使用较小的参数进行快速测试）
    ants_num = 5
    max_iter = 10
    beta = 2
    q0 = 0.1
    
    print(f"数据文件: {file_path}")
    print(f"蚂蚁数量: {ants_num}")
    print(f"最大迭代次数: {max_iter}")
    print("动画将保存到 ./test_animations/ 目录")
    
    # 创建图对象
    graph = VrptwGraph(file_path)
    
    # 创建队列用于传递路径信息
    path_queue_for_figure = Queue()
    
    # 创建保存动画的可视化对象
    save_figure = SaveAnimationFigure(graph.nodes, path_queue_for_figure, 
                                      save_path="./test_animations/")
    
    # 定义算法运行函数
    def run_algorithm():
        print("算法开始运行...")
        basic_aco = BasicACO(graph, ants_num=ants_num, max_iter=max_iter, beta=beta, q0=q0,
                             whether_or_not_to_show_figure=False)
        basic_aco._basic_aco(path_queue_for_figure)
        
        # 发送结束信号
        path_queue_for_figure.put(PathMessage(None, None))
        print("算法运行完成")
    
    # 在新线程中运行算法
    algorithm_thread = Thread(target=run_algorithm)
    algorithm_thread.start()
    
    # 运行并保存动画（保存图片帧但不保存GIF，以便快速测试）
    print("开始保存动画帧...")
    save_figure.run_and_save(save_static=True, save_gif=False)
    
    # 等待算法线程完成
    algorithm_thread.join()
    
    print("测试完成！")
    print("请查看 ./test_animations/ 目录中的保存文件")


if __name__ == '__main__':
    test_save_animation() 