import concurrent.futures
import logging

import numpy as np
import torch
from tqdm import tqdm

# 设置日志记录配置
logging.basicConfig(filename='cs_dissimilarity.log', level=logging.INFO)


def calculate_cs_distance(i, channel, powers, port_num, sc_num):
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
    dCS_i_up = torch.maximum(torch.sum(dCS_i_ele, dim=(-2, -1)) / (port_num * sc_num),
                             torch.zeros_like(dCS_i_ele[:, 0, 0]))

    del dCS_i_ele  # 释放中间结果内存

    return i, dCS_i_up


def calculate_cs_dissimilarity_matrix(channel, max_workers=5):
    """
    计算所有点之间的CS距离，使用分行并行处理
    """
    logging.info("开始计算CS距离")

    # 将数据传输到GPU
    channel = torch.tensor(channel)
    samp_num = channel.shape[0]
    port_num = channel.shape[1]
    sc_num = channel.shape[3]

    output = torch.zeros((samp_num, samp_num), dtype=torch.float32)
    powers = torch.real(torch.einsum("lbmt,lbmt->lbt", channel, torch.conj(channel)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calculate_cs_distance, i, channel, powers, port_num, sc_num): i for i in
                   range(samp_num)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=samp_num, desc="计算CS距离", leave=True):
            i, dCS_i_up = future.result()
            dCS_i = torch.cat([torch.zeros(i), dCS_i_up], 0)
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


if __name__ == '__main__':
    # 配置信息
    ant_num = 64
    port_num = 2
    bs_pos = [0.0, 0.0, 30.0]
    sc_num = 408
    anch_samp_num = 20
    tol_samp_num = 200

    # 锚点位置文件
    anch_pos = np.random.randn(anch_samp_num, 3)  # 示例数据

    # 信道文件
    H = np.random.randn(tol_samp_num, port_num, ant_num, sc_num) + 1j * np.random.randn(tol_samp_num, port_num, ant_num,
                                                                                        sc_num)  # 示例数据

    # 初始化日志记录
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    logging.info("开始计算CS距离")
    distances = calculate_cs_dissimilarity_matrix(H)
    logging.info("CS距离计算完毕，共计算了 {} 对点之间的距离。".format(len(distances)))
