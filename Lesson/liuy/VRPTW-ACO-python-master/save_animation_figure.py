import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Queue as MPQueue
import os
import numpy as np
from PIL import Image
import io


class SaveAnimationFigure:
    def __init__(self, nodes: list, path_queue: MPQueue, save_path="./animations/"):
        """
        带有保存功能的动态可视化类
        
        :param nodes: nodes是各个结点的list，包括depot
        :param path_queue: queue用来存放工作线程计算得到的path
        :param save_path: 保存路径
        """
        self.nodes = nodes
        self.path_queue = path_queue
        self.save_path = save_path
        self.frame_count = 0
        self.frames = []  # 存储所有帧
        
        # 创建保存目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 设置图形参数
        self.figure = plt.figure(figsize=(12, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)
        self._depot_color = 'red'
        self._customer_color = 'steelblue'
        self._line_color = 'darksalmon'
        
        # 设置图形样式
        self.figure_ax.set_xlabel('X坐标', fontsize=12)
        self.figure_ax.set_ylabel('Y坐标', fontsize=12)
        self.figure_ax.grid(True, alpha=0.3)
        
    def _draw_point(self):
        """绘制所有节点"""
        # 画出depot（配送中心）
        self.figure_ax.scatter([self.nodes[0].x], [self.nodes[0].y], 
                              c=self._depot_color, label='配送中心', s=100, marker='s')
        
        # 画出customer（客户点）
        self.figure_ax.scatter(list(node.x for node in self.nodes[1:]),
                              list(node.y for node in self.nodes[1:]), 
                              c=self._customer_color, label='客户点', s=50, marker='o')
        
        # 添加节点标签
        for i, node in enumerate(self.nodes):
            if i == 0:  # depot
                self.figure_ax.annotate(f'D{i}', (node.x, node.y), 
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, color='red', weight='bold')
            else:  # customer
                self.figure_ax.annotate(f'C{i}', (node.x, node.y), 
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=6, color='blue')
        
        self.figure_ax.legend(loc='upper right')
        
    def _save_frame(self, filename):
        """保存当前帧"""
        full_path = os.path.join(self.save_path, filename)
        self.figure.savefig(full_path, dpi=150, bbox_inches='tight')
        print(f"帧已保存: {full_path}")
        
        # 同时保存到内存用于制作GIF
        buffer = io.BytesIO()
        self.figure.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image = Image.open(buffer)
        self.frames.append(image.copy())
        buffer.close()
        
    def _draw_line(self, path, distance, vehicle_num):
        """绘制路径线条"""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        vehicle_paths = []
        current_path = []
        
        # 分割路径为不同车辆的路径
        for i in range(len(path)):
            current_path.append(path[i])
            if i > 0 and path[i] == 0:  # 回到depot
                vehicle_paths.append(current_path[:])
                current_path = [0]  # 开始新的车辆路径
        
        # 绘制每条车辆路径
        for vehicle_idx, vehicle_path in enumerate(vehicle_paths):
            if len(vehicle_path) > 1:
                color = colors[vehicle_idx % len(colors)]
                
                for i in range(1, len(vehicle_path)):
                    x_list = [self.nodes[vehicle_path[i-1]].x, self.nodes[vehicle_path[i]].x]
                    y_list = [self.nodes[vehicle_path[i-1]].y, self.nodes[vehicle_path[i]].y]
                    
                    self.figure_ax.plot(x_list, y_list, color=color, linewidth=2, 
                                       label=f'车辆{vehicle_idx+1}' if i == 1 else "", alpha=0.8)
        
        # 更新标题
        self.figure_ax.set_title(f'VRPTW解决方案 - 总距离: {distance:.2f}, 车辆数: {vehicle_num}', 
                                fontsize=14, fontweight='bold')
        
        # 更新图例
        handles, labels = self.figure_ax.get_legend_handles_labels()
        # 只保留配送中心、客户点和车辆路径的图例
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        self.figure_ax.legend(unique_handles, unique_labels, loc='upper right')
        
    def run_and_save(self, save_static=True, save_gif=True):
        """运行并保存动画"""
        print("开始运行动画并保存...")
        
        # 先绘制出各个结点
        self._draw_point()
        
        # 保存初始帧
        if save_static:
            self._save_frame(f"frame_{self.frame_count:04d}_initial.png")
            self.frame_count += 1
        
        self.figure.show()
        
        # 从队列中读取新的path，进行绘制和保存
        while True:
            if not self.path_queue.empty():
                # 取队列中最新的一个path
                info = self.path_queue.get()
                while not self.path_queue.empty():
                    info = self.path_queue.get()
                
                path, distance, used_vehicle_num = info.get_path_info()
                if path is None:
                    print('[保存动画]: 算法结束，开始生成最终文件')
                    break
                
                # 清除之前的路径线条
                remove_obj = []
                for line in self.figure_ax.lines:
                    remove_obj.append(line)
                for line in remove_obj:
                    line.remove()
                
                # 重新绘制路径
                self._draw_line(path, distance, used_vehicle_num)
                
                # 保存当前帧
                if save_static:
                    filename = f"frame_{self.frame_count:04d}_dist_{distance:.1f}_vehicles_{used_vehicle_num}.png"
                    self._save_frame(filename)
                    self.frame_count += 1
                
                plt.draw()
                plt.pause(0.1)
            else:
                plt.pause(0.1)
        
        # 生成GIF动画
        if save_gif and self.frames:
            self._create_gif()
        
        print(f"动画保存完成！共生成 {self.frame_count} 帧")
        print(f"文件保存在: {self.save_path}")
        
    def _create_gif(self):
        """创建GIF动画"""
        if not self.frames:
            print("没有帧可以创建GIF")
            return
        
        gif_path = os.path.join(self.save_path, "vrptw_optimization_process.gif")
        
        # 为每一帧添加适当的持续时间
        durations = []
        for i in range(len(self.frames)):
            if i == 0:  # 第一帧停留更长时间
                durations.append(2000)
            elif i == len(self.frames) - 1:  # 最后一帧停留更长时间
                durations.append(3000)
            else:
                durations.append(1000)
        
        try:
            self.frames[0].save(
                gif_path,
                save_all=True,
                append_images=self.frames[1:],
                duration=durations,
                loop=0,
                optimize=True
            )
            print(f"GIF动画已保存: {gif_path}")
        except Exception as e:
            print(f"保存GIF时出错: {e}")
            
    def save_final_result(self, path, distance, vehicle_num, filename="final_result.png"):
        """保存最终结果的高质量图片"""
        plt.figure(figsize=(14, 12))
        fig_ax = plt.subplot(1, 1, 1)
        
        # 绘制节点
        fig_ax.scatter([self.nodes[0].x], [self.nodes[0].y], 
                      c='red', label='配送中心', s=150, marker='s', zorder=5)
        fig_ax.scatter(list(node.x for node in self.nodes[1:]),
                      list(node.y for node in self.nodes[1:]), 
                      c='steelblue', label='客户点', s=80, marker='o', zorder=4)
        
        # 添加节点标签
        for i, node in enumerate(self.nodes):
            if i == 0:
                fig_ax.annotate(f'配送中心', (node.x, node.y), 
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=10, color='red', weight='bold')
            else:
                fig_ax.annotate(f'{i}', (node.x, node.y), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='blue')
        
        # 绘制路径
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
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
                    x_list = [self.nodes[vehicle_path[i-1]].x, self.nodes[vehicle_path[i]].x]
                    y_list = [self.nodes[vehicle_path[i-1]].y, self.nodes[vehicle_path[i]].y]
                    
                    fig_ax.plot(x_list, y_list, color=color, linewidth=2.5, 
                               label=f'车辆{vehicle_idx+1}' if i == 1 else "", alpha=0.8, zorder=3)
        
        fig_ax.set_xlabel('X坐标', fontsize=14)
        fig_ax.set_ylabel('Y坐标', fontsize=14)
        fig_ax.set_title(f'VRPTW最终解决方案\n总距离: {distance:.2f}, 使用车辆数: {vehicle_num}', 
                        fontsize=16, fontweight='bold')
        fig_ax.grid(True, alpha=0.3)
        fig_ax.legend(loc='upper right', fontsize=10)
        
        # 保存高质量图片
        final_path = os.path.join(self.save_path, filename)
        plt.savefig(final_path, dpi=300, bbox_inches='tight', format='png')
        print(f"最终结果已保存: {final_path}")
        plt.close() 