import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from mapping import map_embeddings_to_coords
from anchorplt import plt_point_hull
from dataset import read_all_of_huaweicup, tensor_file_name_converter


# 选择三元组 (anchor, positive, negative)
def select_triplets(Dgeo, q, num_triplets):
    """
    从所有样本点中选择出
    input:
        Dgeo : 测地线矩阵，用于给出两点之间的距离
        q    : Q值，用来选择近邻有多近，取值 0-1
        num_triplets : 输出的 triplets 数量
    output:
        triplets : 输出的 triplets 三元组
    """

    L = Dgeo.shape[0]
    idx = int(q * L)
    triplets = []
    for _ in range(num_triplets):
        i = np.random.randint(0, L)
        pos_indices = np.argsort(Dgeo[i, :])[:idx]  # 前qL个最近邻
        neg_indices = np.argsort(Dgeo[i, :])[idx:]  # 后L-qL个远邻
        j = np.random.choice(pos_indices)
        k = np.random.choice(neg_indices)
        triplets.append((i, j, k))
    return triplets


# 定义嵌入网络
class EmbedNet(nn.Module):
    def __init__(self):
        super(EmbedNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(64 * 64 * 408, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)  # 新增的中间层
        self.fc4 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # 新增的中间层
        x = self.fc4(x)  # 输出层
        return x


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

    def embed_to_coords(self, embeds):
        max_distance = np.linalg.norm(anch_pos.max(axis=0) - anch_pos.min(axis=0))
        normalized_embeds = embeds / embeds.norm(dim=1, keepdim=True)
        return normalized_embeds * max_distance


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
        return torch.tensor(self.H[i], dtype=torch.float32), \
            torch.tensor(self.H[j], dtype=torch.float32), \
            torch.tensor(self.H[k], dtype=torch.float32)


def train_triplet_network(triple_net, criterion, optimizer, scheduler, H_combined, d_net, init_q=0.3,
                          num_triplets=20000, num_epochs=20, init_batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用" + str(device) + "进行计算")
    q = init_q
    batch_size = init_batch_size
    for epoch in range(num_epochs):
        triplets = select_triplets(d_net, q, num_triplets)
        triplet_dataset = TripletDataset(H_combined, triplets)
        triplet_loader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True)

        running_loss = 0.0
        triplet_loader_tqdm = tqdm(triplet_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for i, (anchor, positive, negative) in enumerate(triplet_loader_tqdm):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            anchor_out, positive_out, negative_out = triple_net(anchor, positive, negative)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(triple_net.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # 每100个batch打印一次信息
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

    optimizer = optim.Adam(embed_net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch将学习率减半

    # 训练网络
    train_triplet_network(embed_net, criterion, optimizer, scheduler, H_combined, d_geo)

    # 保存网络实例
    torch_net_file = tensor_file_name_converter(1, 1)
    torch.save(embed_net.state_dict(), torch_net_file)
    print("Model saved as " + torch_net_file)
