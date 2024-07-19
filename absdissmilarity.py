import torch


def calculate_abs_dissimilarity_matrix(anch_pos, channel):
    # 将数据传输到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用" + str(device) + "进行计算")
    anch_pos = torch.tensor(anch_pos, device=device)
    channel = torch.tensor(channel, device=device)
