import torch
import numpy as np
import cvxpy as cp
import tritrain
import matplotlib.pyplot as plt
from dataset import read_all_of_huaweicup
from scipy.spatial import ConvexHull


def map_embeddings(embed_net, H_combined, device):
    embeddings = []
    with torch.no_grad():
        for i in range(0, H_combined.shape[0], 32):
            batch = torch.tensor(H_combined[i:i + 32], dtype=torch.float32).to(device)
            embedding = embed_net.embed_net(batch)
            embeddings.append(embedding.cpu().numpy())
    return np.vstack(embeddings)


def map_embeddings_to_coords(embeddings, anch_pos):
    """
    将嵌入向量映射到绝对坐标，并添加边界条件。

    参数:
    embeddings (numpy.ndarray): 要映射的嵌入向量 (N, D)，其中 N 是嵌入向量的数量，D 是嵌入向量的维度。
    anch_pos (numpy.ndarray): 具有绝对坐标的点 (M, 3)，其中 M 是已知嵌入向量的数量，anch_pos[:, 0] 是索引，anch_pos[:, 1:] 是绝对坐标。

    返回:
    numpy.ndarray: 映射后的绝对坐标 (N, 2)。
    """
    # 从 anch_pos 中提取已知的嵌入向量和绝对坐标
    print("Start mapping embeddings to coords through CVX toolbox")
    indices = anch_pos[:, 0].astype(int)
    known_embeds = embeddings[indices]
    known_coords = anch_pos[:, 1:]

    print("Known indices:", indices[:5])
    print("Known embeddings (first 5):", known_embeds[:5])
    print("Known coordinates (first 5):", known_coords[:5])

    # 获取嵌入向量的维度和绝对坐标的维度
    embedding_dim = known_embeds.shape[1]
    coords_dim = known_coords.shape[1]

    # 定义优化变量
    A = cp.Variable((embedding_dim, coords_dim))  # 线性变换矩阵
    b = cp.Variable((1, coords_dim))  # 平移向量

    # 定义目标函数：最小化变换后的嵌入向量与绝对坐标之间的误差
    objective = cp.Minimize(cp.sum_squares(known_embeds @ A + b - known_coords))

    # 初步定义无约束优化问题
    problem = cp.Problem(objective)

    # 求解无约束优化问题
    problem.solve()
    print("Initial problem status:", problem.status)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Initial optimization did not converge to an optimal solution.")
        return None

    # 获取初步优化结果
    A_opt = A.value
    b_opt = b.value

    print("Initial Optimal A:", A_opt)
    print("Initial Optimal b:", b_opt)

    # 将嵌入向量转换为绝对坐标
    mapped_coords = embeddings @ A_opt + b_opt
    #
    # # 计算凸包边界
    # hull = calculate_expanded_hull(known_coords, 3)
    # print("Convex hull equations:", hull.equations)
    # plt_anchor_hull(anch_pos, hull)
    #
    # # 添加凸包边界约束并重新定义优化问题
    # constraints = []
    # for simplex in hull.equations:
    #     constraints.append(mapped_coords @ simplex[:-1] <= simplex[-1])
    #     print("Added constraint:", simplex)
    #
    # # 定义带约束的优化问题
    # problem = cp.Problem(objective, constraints)
    #
    # # 求解带约束的优化问题
    # problem.solve()
    #
    # # 检查优化问题状态
    # print("Problem status with constraints:", problem.status)
    # if problem.status not in ["optimal", "optimal_inaccurate"]:
    #     print("Optimization with constraints did not converge to an optimal solution.")
    #     return None
    #
    # # 获取带约束优化结果
    # A_opt = A.value
    # b_opt = b.value
    #
    # print("Optimal A with constraints:", A_opt)
    # print("Optimal b with constraints:", b_opt)
    #
    # # 将嵌入向量转换为绝对坐标
    # mapped_coords = embeddings @ A_opt + b_opt

    return mapped_coords


# 示例使用
if __name__ == '__main__':
    # 假设网络给出的嵌入向量和 anch_pos 的定义已经在您的代码中
    num_points = 20000
    embedding_dim = 64
    bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num, anch_pos, H, d_geo = read_all_of_huaweicup(1, 1)

    H_real = H.real
    H_imag = H.imag
    H_combined = np.stack((H_real, H_imag), axis=2)
    H_combined = H_combined.reshape(tol_samp_num, 4, ant_num, sc_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 在需要使用模型的地方加载模型
    embed_net_loaded = tritrain.TripletNet().to(device)
    embed_net_loaded.load_state_dict(torch.load('./CompetitionData1/Round1NET11721287896.9979763.pth'))
    embed_net_loaded.eval()  # 切换到评估模式

    print("Model loaded from Round1NET11721287896.9979763.pth")
    embeddings = map_embeddings(embed_net_loaded, H_combined, device)

    # 调用函数
    mapped_coords = map_embeddings_to_coords(embeddings, anch_pos)

    # 输出前五个映射结果
    print("Mapped Coordinates for the first 5 points:")
    print(mapped_coords[:5])

    # 验证转换效果
    if mapped_coords is not None:
        print("Original Known Coordinates:")
        print(anch_pos[:5, 1:])
        print("Mapped Coordinates for Known Embeddings:")
        mapped_known_coords = mapped_coords[anch_pos[:5, 0].astype(int)]
        print(mapped_known_coords[:5])
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.scatter(mapped_coords[:, 0], mapped_coords[:, 1], s=10, label='Mapped Points')
        plt.scatter(anch_pos[:, 1], anch_pos[:, 2], s=50, label='Anchor Points', marker='x')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Mapped Coordinates vs Anchor Points')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 计算差值
        indices = anch_pos[:, 0].astype(int)
        real_coords = anch_pos[:, 1:]
        diff = mapped_coords[indices] - real_coords
        # 计算所有锚点偏差的总和
        total_diff_sum = np.sum(np.abs(diff))  # 可以改为 np.linalg.norm(diff) 求范数

        print("Differences between Mapped and Real Coordinates:", diff[:5])
        print("Total deviation of anchor points:", total_diff_sum)
    else:
        print("Mapping failed.")
