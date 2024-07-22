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


def map_embeddings_to_coords(embeddings, anch_pos, device):
    """
    将嵌入向量映射到绝对坐标，并添加边界条件。

    参数:
    embeddings (numpy.ndarray): 要映射的嵌入向量 (N, D)，其中 N 是嵌入向量的数量，D 是嵌入向量的维度。
    anch_pos (numpy.ndarray): 具有绝对坐标的点 (M, 3)，其中 M 是已知嵌入向量的数量，anch_pos[:, 0] 是索引，anch_pos[:, 1:] 是绝对坐标。

    返回:
    numpy.ndarray: 映射后的绝对坐标 (N, 2)。
    """
    # 从 anch_pos 中提取已知的嵌入向量和绝对坐标
    print("Start mapping embeddings to coords through PyTorch")
    indices = anch_pos[:, 0].astype(int)
    known_embeds = torch.tensor(embeddings[indices], dtype=torch.float32, device=device)
    known_coords = torch.tensor(anch_pos[:, 1:], dtype=torch.float32, device=device)

    print("Known indices:", indices[:5])
    print("Known embeddings (first 5):", known_embeds[:5])
    print("Known coordinates (first 5):", known_coords[:5])

    # 获取嵌入向量的维度和绝对坐标的维度
    embedding_dim = known_embeds.shape[1]
    coords_dim = known_coords.shape[1]

    # 定义优化变量
    A = torch.randn((embedding_dim, coords_dim), requires_grad=True, device=device)  # 线性变换矩阵
    b = torch.randn((1, coords_dim), requires_grad=True, device=device)  # 平移向量

    # 定义优化器
    optimizer = torch.optim.Adam([A, b], lr=0.01)

    # 优化迭代
    num_iterations = 20000
    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss = torch.sum((known_embeds @ A + b - known_coords) ** 2)
        loss.backward()
        optimizer.step()
        if _ % 2000 == 0:
            print(f"Iteration {_}: Loss = {loss.item()}")

    # 获取优化结果
    A_opt = A.detach().cpu().numpy()
    b_opt = b.detach().cpu().numpy()

    print("Optimal A:", A_opt)
    print("Optimal b:", b_opt)

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

