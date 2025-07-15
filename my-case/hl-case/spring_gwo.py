import numpy as np

# 目标函数：弹簧重量
def spring_weight(x):
    d, D, N = x
    return (np.pi ** 2) * D * (d ** 2) * N / 4  # 计算弹簧重量
    

# 约束条件
def constraint(x):
    d, D, N = x
    # 常数
    F = 1000  # 载荷N
    G = 80000  # 剪切模量MPa
    L_max = 14  # 最大自由长度mm
    tau_max = 1890  # 最大剪切应力MPa
    delta_max = 6  # 最大挠度mm

    # 计算剪切应力
    tau = (8 * F * D) / (np.pi * d ** 3)
    # 计算挠度
    delta = (8 * F * D ** 3 * N) / (G * d ** 4)
    # 自由长度
    L = (N + 1) * d

    # 约束列表，负值为满足，正值为违反
    return [
        # 这里就是限制条件，限制了弹簧的剪切应力，挠度，自由长度，d，D，N的取值范围，没有其他限制了
        tau - tau_max,      # 剪切应力约束
        delta - delta_max,  # 挠度约束
        L - L_max,          # 自由长度约束
        0.05 - d,           # d下限
        d - 2.0,            # d上限
        0.25 - D,           # D下限
        D - 1.3,            # D上限
        2.0 - N,            # N下限
        N - 15.0            # N上限
    ]

# 适应度函数（带惩罚项）
def fitness(x):
    penalty = 0
    cons = constraint(x)
    for c in cons:
        if c > 0:
            penalty += 1e6 * c  # 违反约束时加大惩罚
    return spring_weight(x) + penalty

# 灰狼优化算法
def gwo(obj_func, dim, lb, ub, max_iter=100, pop_size=20):
    # 初始化狼群
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")
    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")
    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")

    # 初始化种群
    Positions = np.random.uniform(lb, ub, (pop_size, dim))

    # 记录历史
    history = []

    for l in range(max_iter):
        for i in range(pop_size):
            # 越界修正
            Positions[i] = np.clip(Positions[i], lb, ub)
            # 计算适应度
            fitness_val = obj_func(Positions[i])

            # 更新Alpha, Beta, Delta
            if fitness_val < Alpha_score:
                Alpha_score = fitness_val
                Alpha_pos = Positions[i].copy()
            elif fitness_val < Beta_score:
                Beta_score = fitness_val
                Beta_pos = Positions[i].copy()
            elif fitness_val < Delta_score:
                Delta_score = fitness_val
                Delta_pos = Positions[i].copy()

        a = 2 - l * (2 / max_iter)  # a从2线性减到0

        for i in range(pop_size):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                # 更新位置
                Positions[i, j] = (X1 + X2 + X3) / 3

        # 记录每一代的最优解
        history.append((Alpha_pos.copy(), Alpha_score))

    return Alpha_pos, Alpha_score, history

if __name__ == "__main__":
    # 设计变量维度
    dim = 3
    # 变量下限
    lb = np.array([0.05, 0.25, 2.0])
    # 变量上限
    ub = np.array([2.0, 1.3, 15.0])

    # 运行次数
    test_times = 5  # 可根据需要修改

    results = []
    for i in range(test_times):
        best_x, best_score, history = gwo(fitness, dim, lb, ub, max_iter=100, pop_size=20)
        results.append((best_x, best_score))
        print(f"第{i+1}组最优参数: d={best_x[0]:.4f}, D={best_x[1]:.4f}, N={best_x[2]:.4f}，最小重量={best_score:.4f}")

    print("\n所有结果对比：")
    for idx, (x, score) in enumerate(results):
        print(f"第{idx+1}组: d={x[0]:.4f}, D={x[1]:.4f}, N={x[2]:.4f}，最小重量={score:.4f}")
