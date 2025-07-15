# spring_visual.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 增加matplotlib识别中文
import matplotlib
matplotlib.rcParams['font.family'] = ['SimHei']

# 绘制弹簧的3D模型
def plot_spring(ax, d, D, N, color='b', label=None):
    # d: 钢丝直径, D: 线圈直径, N: 有效圈数
    coils = int(np.round(N))
    points_per_coil = 50
    total_points = coils * points_per_coil
    theta = np.linspace(0, 2 * np.pi * coils, total_points)
    x = (D / 2) * np.cos(theta)
    y = (D / 2) * np.sin(theta)
    z = np.linspace(0, (N + 1) * d, total_points)
    ax.plot3D(x, y, z, color=color, label=label, linewidth=2)
    # 绘制弹簧两端
    ax.scatter([x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]], color='k', s=30)

# 绘制参数变化曲线
def plot_param_history(history):
    d_list = [h[0][0] for h in history]
    D_list = [h[0][1] for h in history]
    N_list = [h[0][2] for h in history]
    plt.figure()
    plt.plot(d_list, label='d (钢丝直径)')
    plt.plot(D_list, label='D (线圈直径)')
    plt.plot(N_list, label='N (线圈数)')
    plt.xlabel('迭代次数')
    plt.ylabel('参数值')
    plt.legend()
    plt.title('弹簧参数变化曲线')
    plt.tight_layout()
    plt.show()