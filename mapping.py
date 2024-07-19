import cvxpy as cp
import numpy as np
import torch


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

    return mapped_coords


def save_coords(mapped_coords, anch_pos):
    """
    将所有样本的映射坐标保存为列表，其中锚点直接用真实坐标。

    参数:
    mapped_coords (numpy.ndarray): 映射后的绝对坐标 (N, 2)。
    anch_pos (numpy.ndarray): 具有绝对坐标的点 (M, 3)，其中 M 是已知嵌入向量的数量，anch_pos[:, 0] 是索引，anch_pos[:, 1:] 是绝对坐标。

    返回:
    list: 保存所有样本坐标的列表。
    """
    indices = anch_pos[:, 0].astype(int)
    xy_list = mapped_coords.tolist()

    for idx in indices:
        xy_list[idx] = anch_pos[anch_pos[:, 0] == idx, 1:].tolist()[0]

    return xy_list


def save_coords_to_txt(xy_list, filename):
    with open(filename, 'w') as f:
        for coord in xy_list:
            f.write(f"{coord[0]} {coord[1]}\n")
