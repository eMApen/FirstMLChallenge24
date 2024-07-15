import logging
import os
import time

import numpy as np

import dataset
from algorithm import calcLoc

if __name__ == "__main__":
    print("<<< Welcome to 2024 Wireless Algorithm Contest! >>>\n")
    ## 不同轮次的输入数据可放在不同文件夹中便于管理，这里用户可以自定义
    PathSet = {0: "./Test", 1: "./CompetitionData1", 2: "./CompetitionData2", 3: "./CompetitionData3"}
    PrefixSet = {0: "Round0", 1: "Round1", 2: "Round2", 3: "Round3"}

    Ridx = 1  # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]

    # 查找文件夹中包含的所有比赛/测试数据文件，非本轮次数据请不要放在目标文件夹中
    files = os.listdir(PathRaw)
    names = []
    for f in sorted(files):
        if f.find('CfgData') != -1 and f.endswith('.txt'):
            names.append(f.split('CfgData')[-1].split('.txt')[0])

    ## 创建对象并处理
    for na in names:
        FileIdx = int(na)
        print('Processing Round ' + str(Ridx) + ' Case ' + str(na))

        # 读取配置文件 RoundYCfgDataX.txt
        print('Loading configuration data file')
        cfg_path = PathRaw + '/' + Prefix + 'CfgData' + na + '.txt'
        bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num = dataset.read_cfg_file(cfg_path)

        # 读取锚点位置文件 RoundYInputPosX.txt
        print('Loading input position file')
        anch_pos_path = PathRaw + '/' + Prefix + 'InputPos' + na + '.txt'
        anch_pos = dataset.read_anch_file(anch_pos_path, anch_samp_num)

        # 读取信道文件 RoundYInputDataX.txt
        slice_samp_num = 1000  # 每个切片读取的数量
        slice_num = int(tol_samp_num / slice_samp_num)  # 切片数量
        csi_path = PathRaw + '/' + Prefix + 'InputData' + na + '.txt'

        # H = []
        # for slice_idx in range(slice_num):  # range(slice_num): # 分切片循环读取信道数据
        #     print('Loading input CSI data of slice ' + str(slice_idx))
        #     slice_lines = dataset.read_slice_of_file(csi_path, slice_idx * slice_samp_num, (slice_idx + 1) * slice_samp_num)
        #     Htmp = np.loadtxt(slice_lines)
        #     Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
        #     Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
        #     Htmp = np.transpose(Htmp, (0, 3, 2, 1))
        #     if np.size(H) == 0:
        #         H = Htmp
        #     else:
        #         H = np.concatenate((H, Htmp), axis=0)
        # H = H.astype(np.complex64)  # 默认读取为Complex128精度，转换为Complex64降低存储开销

        csi_file = PathRaw + '/' + Prefix + 'InputData' + na + '.npy'
        # np.save(csi_file, H)  # 首次读取后可存储为npy格式的数据文件
        H = np.load(csi_file)  # 后续可以直接load数据文件

        tStart = time.perf_counter()

        logging.info("开始计算ADP距离")
        distances = calADP.calculate_adp_distances_parallel(H, tol_samp_num)
        logging.info("ADP距离计算完毕，共计算了 {} 对点之间的距离。".format(len(distances)))

        # 计算并输出定位位置
        print('Calculating localization results')

        # # 示例用法
        # adp_matrix = calADP.compute_adp_with_weighting(H, anch_pos, bs_pos, sc_num, ant_num, port_num, anch_samp_num,
        #                                                tol_samp_num, 0, 0)
        # adp_file = PathRaw + '/' + Prefix + 'ADPData' + na + '.npy'
        # np.save(adp_file, adp_matrix)  # 首次读取后可存储为npy格式的数据文件

        result = calcLoc(H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num)  # 核心定位功能函数，由参赛者实现

        # 回填锚点位置信息
        for idx in range(anch_samp_num):
            rowIdx = int(anch_pos[idx][0] - 1)
            result[rowIdx] = np.array([anch_pos[idx][1], anch_pos[idx][2]])

        # 输出结果：各位参赛者注意输出值的精度
        print('Writing output position file')
        with open(PathRaw + '/' + Prefix + 'OutputPos' + na + '.txt', 'w') as f:
            np.savetxt(f, result, fmt='%.4f %.4f')
        # 统计时间
        tEnd = time.perf_counter()
        print("Total time consuming = {}s\n\n".format(round(tEnd - tStart, 3)))
