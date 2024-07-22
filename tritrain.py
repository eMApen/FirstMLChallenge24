import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from mapping import map_embeddings_to_coords
from anchorplt import plt_point_hull
from dataset import tensor_file_name_converter, read_all_of_huaweicup

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 选择三元组 (anchor, positive, negative)
def select_triplets(Dgeo, q, num_triplets, device):
    L = Dgeo.shape[0]
    idx = int(q * L)
    triplets = []
    for _ in range(num_triplets):
        i = torch.randint(0, L, (1,), device=device).item()
        sorted_indices = torch.argsort(Dgeo[i])
        pos_indices = sorted_indices[:idx]
        neg_indices = sorted_indices[idx:]
        j = pos_indices[torch.randint(0, len(pos_indices), (1,), device=device).item()].item()
        k = neg_indices[torch.randint(0, len(neg_indices), (1,), device=device).item()].item()
        triplets.append((i, j, k))
    return triplets


# 定义嵌入网络
class EmbedNet(nn.Module):
    def __init__(self):
        super(EmbedNet, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 64 * 408, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出2维坐标
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展开成一维向量
        return self.fc_layers(x)


# 定义三元组网络
class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.embed_net = EmbedNet()

    def forward(self, anchor, positive, negative):
        anchor_out = self.embed_net(anchor)
        positive_out = self.embed_net(positive)
        negative_out = self.embed_net(negative)
        return anchor_out, positive_out, negative_out

    def embed_to_coords(self, embeds, A, b, anch_pos):
        anch_pos = anch_pos[:, 1:]  # anch_pos 已经在 GPU 上，无需再转换
        max_distance = torch.norm(anch_pos.max(dim=0)[0] - anch_pos.min(dim=0)[0])
        normalized_embeds = embeds / embeds.norm(dim=1, keepdim=True)
        return normalized_embeds @ A + b


# 定义三元组损失函数
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(losses)


# 定义数据集类
class TripletDataset(Dataset):
    def __init__(self, H, triplets):
        self.H = H
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        i, j, k = self.triplets[idx]
        return self.H[i].clone().detach().float(), \
            self.H[j].clone().detach().float(), \
            self.H[k].clone().detach().float()


def compute_optimal_A_b(H_combined, known_coords):
    # 计算 A 和 b 的优化值
    # 计算映射矩阵 A 和偏移量 b
    # 这里可以使用类似最小二乘法的方法计算 A 和 b
    # 假设我们用简单的线性回归作为示例
    # 可以根据实际需求使用更复杂的方法
    A = torch.linalg.lstsq(H_combined, known_coords, rcond=None).solution
    b = torch.mean(known_coords - H_combined @ A, dim=0, keepdim=True)

    return A, b


def train_triplet_network(triple_net, criterion, optimizer, scheduler, H_combined, d_net, anch_pos, init_q=0.3,
                          num_triplets=200000, num_epochs=30, init_batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用" + str(device) + "进行计算")
    q = init_q
    batch_size = init_batch_size

    # 将 anch_pos 转换为 GPU 上的张量
    anch_pos = torch.tensor(anch_pos, dtype=torch.float32, device=device)
    H_combined = torch.tensor(H_combined, dtype=torch.float32, device=device)
    d_net = torch.tensor(d_net, dtype=torch.float32, device=device)

    # 初始映射矩阵和偏移向量
    embedding_dim = 2
    coords_dim = 2
    A = torch.randn((embedding_dim, coords_dim), requires_grad=True, device=device)  # 线性变换矩阵
    b = torch.randn((1, coords_dim), requires_grad=True, device=device)  # 平移向量
    map_optimizer = optim.Adam([A, b], lr=0.01)

    for epoch in range(num_epochs):
        triplets = select_triplets(d_net, q, num_triplets, device)
        triplet_dataset = TripletDataset(H_combined, triplets)
        triplet_loader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True)

        running_loss = 0.0
        triplet_loader_tqdm = tqdm(triplet_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for i, (anchor, positive, negative) in enumerate(triplet_loader_tqdm):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            map_optimizer.zero_grad()

            anchor_out, positive_out, negative_out = triple_net(anchor, positive, negative)
            triplet_loss = criterion(anchor_out, positive_out, negative_out)

            # # 计算绝对误差
            # indices = anch_pos[:, 0].long().to(device)
            # known_embeds = triple_net.embed_net(H_combined[indices])
            # known_coords = anch_pos[:, 1:]
            # mapped_coords = triple_net.embed_to_coords(known_embeds, A, b, known_coords)
            # abs_error_loss = torch.sum((mapped_coords - known_coords) ** 2)/indices.shape[0]

            # 联合损失
            # total_loss = triplet_loss + abs_error_loss
            total_loss = triplet_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(triple_net.parameters(), max_norm=1.0)
            optimizer.step()
            map_optimizer.step()

            running_loss += total_loss.item()
            if i % 1000 == 999:  # 每100个batch打印一次信息
                tqdm.write(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}], Q: {q :.4f}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        scheduler.step()  # 每个 epoch 结束后更新学习率
        print(f'Epoch [{epoch + 1}/{num_epochs}] Completed, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        q = max(0.01, q - (init_q - 0.02) / num_epochs)  # 每个 epoch 结束后减小 q 值

        # 动态调整批量大小
        batch_size = min(128, batch_size + 8)  # 增加批量大小，但不超过128
        del triplets


if __name__ == '__main__':
    print("<<< Welcome to 2024 Wireless Algorithm Contest! >>>\n")

    bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num, anch_pos, H, d_geo = read_all_of_huaweicup(1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Selected device " + str(device))
    plt_point_hull(anch_pos)

    embed_net = TripletNet().to(device)

    # 初始化损失函数，给入参数 margin
    margin = 1.0
    criterion = TripletLoss(margin).to(device)

    H_real = H.real
    H_imag = H.imag
    H_combined = np.stack((H_real, H_imag), axis=2)
    H_combined = H_combined.reshape(tol_samp_num, 4, ant_num, sc_num)

    optimizer = optim.Adam(embed_net.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch将学习率减半

    train_triplet_network(embed_net, criterion, optimizer, scheduler, H_combined, d_geo, anch_pos)

    # 保存网络实例
    torch_net_file = tensor_file_name_converter(1, 1)
    torch.save(embed_net.state_dict(), torch_net_file)
    print("Model saved as " + torch_net_file)
