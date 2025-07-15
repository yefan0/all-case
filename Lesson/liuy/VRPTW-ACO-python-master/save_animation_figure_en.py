import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Queue as MPQueue
import os
import numpy as np
from PIL import Image
import io


class SaveAnimationFigureEN:
    def __init__(self, nodes: list, path_queue: MPQueue, save_path="./animations/"):
        """
        Animation saving class with English labels
        
        :param nodes: list of nodes including depot
        :param path_queue: queue for path messages
        :param save_path: save directory path
        """
        self.nodes = nodes
        self.path_queue = path_queue
        self.save_path = save_path
        self.frame_count = 0
        self.frames = []  # Store all frames
        
        # Create save directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Set figure parameters
        self.figure = plt.figure(figsize=(12, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)
        self._depot_color = 'red'
        self._customer_color = 'steelblue'
        self._line_color = 'darksalmon'
        
        # Set figure style
        self.figure_ax.set_xlabel('X Coordinate', fontsize=12)
        self.figure_ax.set_ylabel('Y Coordinate', fontsize=12)
        self.figure_ax.grid(True, alpha=0.3)
        
    def _draw_point(self):
        """Draw all nodes"""
        # Draw depot
        self.figure_ax.scatter([self.nodes[0].x], [self.nodes[0].y], 
                              c=self._depot_color, label='Depot', s=100, marker='s')
        
        # Draw customers
        self.figure_ax.scatter(list(node.x for node in self.nodes[1:]),
                              list(node.y for node in self.nodes[1:]), 
                              c=self._customer_color, label='Customers', s=50, marker='o')
        
        # Add node labels
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
        """Save current frame"""
        full_path = os.path.join(self.save_path, filename)
        self.figure.savefig(full_path, dpi=150, bbox_inches='tight')
        print(f"Frame saved: {full_path}")
        
        # Also save to memory for GIF creation
        buffer = io.BytesIO()
        self.figure.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image = Image.open(buffer)
        self.frames.append(image.copy())
        buffer.close()
        
    def _draw_line(self, path, distance, vehicle_num):
        """Draw path lines"""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        vehicle_paths = []
        current_path = []
        
        # Split path into different vehicle routes
        for i in range(len(path)):
            current_path.append(path[i])
            if i > 0 and path[i] == 0:  # Return to depot
                vehicle_paths.append(current_path[:])
                current_path = [0]  # Start new vehicle path
        
        # Draw each vehicle path
        for vehicle_idx, vehicle_path in enumerate(vehicle_paths):
            if len(vehicle_path) > 1:
                color = colors[vehicle_idx % len(colors)]
                
                for i in range(1, len(vehicle_path)):
                    x_list = [self.nodes[vehicle_path[i-1]].x, self.nodes[vehicle_path[i]].x]
                    y_list = [self.nodes[vehicle_path[i-1]].y, self.nodes[vehicle_path[i]].y]
                    
                    self.figure_ax.plot(x_list, y_list, color=color, linewidth=2, 
                                       label=f'Vehicle {vehicle_idx+1}' if i == 1 else "", alpha=0.8)
        
        # Update title
        self.figure_ax.set_title(f'VRPTW Solution - Distance: {distance:.2f}, Vehicles: {vehicle_num}', 
                                fontsize=14, fontweight='bold')
        
        # Update legend
        handles, labels = self.figure_ax.get_legend_handles_labels()
        # Keep only unique labels
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        self.figure_ax.legend(unique_handles, unique_labels, loc='upper right')
        
    def run_and_save(self, save_static=True, save_gif=True):
        """Run and save animation"""
        print("Starting animation and saving...")
        
        # First draw all nodes
        self._draw_point()
        
        # Save initial frame
        if save_static:
            self._save_frame(f"frame_{self.frame_count:04d}_initial.png")
            self.frame_count += 1
        
        self.figure.show()
        
        # Read new paths from queue and draw/save
        while True:
            if not self.path_queue.empty():
                # Get latest path from queue
                info = self.path_queue.get()
                while not self.path_queue.empty():
                    info = self.path_queue.get()
                
                path, distance, used_vehicle_num = info.get_path_info()
                if path is None:
                    print('[Save Animation]: Algorithm finished, generating final files')
                    break
                
                # Remove previous path lines
                remove_obj = []
                for line in self.figure_ax.lines:
                    remove_obj.append(line)
                for line in remove_obj:
                    line.remove()
                
                # Redraw path
                self._draw_line(path, distance, used_vehicle_num)
                
                # Save current frame
                if save_static:
                    filename = f"frame_{self.frame_count:04d}_dist_{distance:.1f}_vehicles_{used_vehicle_num}.png"
                    self._save_frame(filename)
                    self.frame_count += 1
                
                plt.draw()
                plt.pause(0.1)
            else:
                plt.pause(0.1)
        
        # Generate GIF animation
        if save_gif and self.frames:
            self._create_gif()
        
        print(f"Animation saved! Generated {self.frame_count} frames")
        print(f"Files saved in: {self.save_path}")
        
    def _create_gif(self):
        """Create GIF animation"""
        if not self.frames:
            print("No frames to create GIF")
            return
        
        gif_path = os.path.join(self.save_path, "vrptw_optimization_process.gif")
        
        # Add appropriate duration for each frame
        durations = []
        for i in range(len(self.frames)):
            if i == 0:  # First frame stays longer
                durations.append(2000)
            elif i == len(self.frames) - 1:  # Last frame stays longer
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
            print(f"GIF animation saved: {gif_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
            
    def save_final_result(self, path, distance, vehicle_num, filename="final_result.png"):
        """Save high-quality final result image"""
        plt.figure(figsize=(14, 12))
        fig_ax = plt.subplot(1, 1, 1)
        
        # Draw nodes
        fig_ax.scatter([self.nodes[0].x], [self.nodes[0].y], 
                      c='red', label='Depot', s=150, marker='s', zorder=5)
        fig_ax.scatter(list(node.x for node in self.nodes[1:]),
                      list(node.y for node in self.nodes[1:]), 
                      c='steelblue', label='Customers', s=80, marker='o', zorder=4)
        
        # Add node labels
        for i, node in enumerate(self.nodes):
            if i == 0:
                fig_ax.annotate(f'Depot', (node.x, node.y), 
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=10, color='red', weight='bold')
            else:
                fig_ax.annotate(f'{i}', (node.x, node.y), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='blue')
        
        # Draw paths
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
                               label=f'Vehicle {vehicle_idx+1}' if i == 1 else "", alpha=0.8, zorder=3)
        
        fig_ax.set_xlabel('X Coordinate', fontsize=14)
        fig_ax.set_ylabel('Y Coordinate', fontsize=14)
        fig_ax.set_title(f'VRPTW Final Solution\nTotal Distance: {distance:.2f}, Vehicles Used: {vehicle_num}', 
                        fontsize=16, fontweight='bold')
        fig_ax.grid(True, alpha=0.3)
        fig_ax.legend(loc='upper right', fontsize=10)
        
        # Save high-quality image
        final_path = os.path.join(self.save_path, filename)
        plt.savefig(final_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Final result saved: {final_path}")
        plt.close() 