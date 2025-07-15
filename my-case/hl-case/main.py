# main.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from spring_gwo import gwo, fitness
from spring_visual import plot_spring, plot_param_history

# 参数范围
lb = np.array([0.05, 0.25, 2.0])  # d, D, N下限
ub = np.array([2.0, 1.3, 15.0])   # d, D, N上限

# 运行GWO优化
best_pos, best_score, history = gwo(fitness, 3, lb, ub, max_iter=50, pop_size=20)

print("最优参数：d=%.4f, D=%.4f, N=%.4f" % (best_pos[0], best_pos[1], best_pos[2]))
print("最小弹簧重量：%.4f" % best_score)

# 绘制参数变化曲线
plot_param_history(history)

# 优化前后弹簧对比
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 初始参数
init_params = history[0][0]
plot_spring(ax, *init_params, color='r', label='优化前')
# 最优参数
plot_spring(ax, *best_pos, color='g', label='优化后')
ax.set_title('弹簧优化前后对比')
ax.legend()
plt.show()

# 动态动画
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.cla()
    params = history[frame][0]
    plot_spring(ax, *params, color='b')
    ax.set_title(f'迭代 {frame+1}')
    ax.set_xlim(-ub[1]/2, ub[1]/2)
    ax.set_ylim(-ub[1]/2, ub[1]/2)
    ax.set_zlim(0, (ub[2]+1)*ub[0])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=200)
ani.save('spring_optimization.gif', writer='pillow')
plt.show()