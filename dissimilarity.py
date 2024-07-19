import concurrent.futures
import logging

import torch
from tqdm import tqdm

# 设置日志记录配置
logging.basicConfig(filename='cs_dissimilarity.log', level=logging.INFO)


def calculate_cs_distance(i, channel, powers):
    """
    计算第 i 行的CS距离
    """
    h_i = channel[i, :, :, :].unsqueeze(0)  # 增加维度以便进行广播
    w_i = channel[i:, :, :, :]

    numerator = torch.real(torch.abs(torch.einsum("lbmt,lbmt->lbt", torch.conj(h_i), w_i)) ** 2)
    del h_i, w_i  # 释放中间结果内存
    denominator = powers[i] * powers[i:]
    dCS_i_ele = 1 - torch.div(numerator, denominator)
    del numerator, denominator  # 释放中间结果内存
    dCS_i_up = torch.maximum(torch.sum(dCS_i_ele, dim=(-2, -1)),
                             torch.zeros_like(dCS_i_ele[:, 0, 0]))

    del dCS_i_ele  # 释放中间结果内存

    return i, dCS_i_up


def calculate_cs_dissimilarity_matrix(channel, max_workers=5):
    """
    计算所有点之间的CS距离，使用分行并行处理
    """
    logging.info("开始计算CS距离")

    # 将数据传输到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用" + str(device) + "进行计算")
    channel = torch.tensor(channel, device=device)
    samp_num = channel.shape[0]
    port_num = channel.shape[1]
    sc_num = channel.shape[3]

    output = torch.zeros((samp_num, samp_num), dtype=torch.float32, device=device)
    powers = torch.real(torch.einsum("lbmt,lbmt->lbt", torch.conj(channel), channel))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calculate_cs_distance, i, channel, powers): i for i in
                   range(samp_num)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=samp_num, desc="计算CS距离", leave=True):
            i, dCS_i_up = future.result()
            dCS_i = torch.cat([torch.zeros(i, device=device), dCS_i_up], 0)
            del dCS_i_up
            output[i] = dCS_i

            # 打印或记录当前output矩阵的状态
            if i == 5:  # 示例：仅打印第5行的结果
                torch.set_printoptions(precision=5, threshold=10, edgeitems=4, linewidth=200)
                tqdm.write(f"第 {i} 行计算后output矩阵的状态:\n{output}")

    dCS_upper_matrix = output.clone()

    logging.info("CS距离计算完毕，共计算了 {} 对点之间的距离。".format(samp_num ** 2))

    dCS_matrix = dCS_upper_matrix + torch.transpose(dCS_upper_matrix, 0, 1)

    # 转化为np格式，进行保存
    dCS_np = dCS_matrix.cpu().numpy()

    return dCS_np

