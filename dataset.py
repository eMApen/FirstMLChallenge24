import itertools
import os
import time

import numpy as np
from tqdm import tqdm

## 不同轮次的输入数据可放在不同文件夹中便于管理，这里用户可以自定义
PathSet = {0: "./Test", 1: "./CompetitionData1", 2: "./CompetitionData2", 3: "./CompetitionData3"}
PrefixSet = {0: "Round0", 1: "Round1", 2: "Round2", 3: "Round3"}


# 读取配置文件函数

def read_cfg_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        line_fmt = [line.rstrip('\n').split(' ') for line in lines]

    info = line_fmt
    bs_pos = list([float(info[0][0]), float(info[0][1]), float(info[0][2])])
    tol_samp_num = int(info[1][0])
    anch_samp_num = int(info[2][0])
    port_num = int(info[3][0])
    ant_num = int(info[4][0])
    sc_num = int(info[5][0])
    return bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num


# 读取锚点位置文件函数

def read_anch_file(file_path, anch_samp_num):
    anch_pos = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        line_fmt = [line.rstrip('\n').split(' ') for line in lines]
    for line in line_fmt:
        tmp = np.array([int(line[0]), float(line[1]), float(line[2])])
        if np.size(anch_pos) == 0:
            anch_pos = tmp
        else:
            anch_pos = np.vstack((anch_pos, tmp))
    return anch_pos


# 切片读取信道文件函数

def read_slice_of_file(file_path, start, end):
    with open(file_path, 'r') as file:
        # 使用itertools.islice进行切片处理
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines


def read_channel_csi(inputdata_path, tol_samp_num, sc_num, ant_num, port_num):
    slice_samp_num = 1000  # 切片样本数量
    slice_num = int(tol_samp_num / slice_samp_num)  # 切片数量
    # 切片读取RoundYInputDataX.txt信道信息
    H = []
    for slice_idx in tqdm(range(slice_num)):
        slice_lines = read_slice_of_file(inputdata_path, slice_idx * slice_samp_num, (slice_idx + 1) * slice_samp_num)
        Htmp = np.loadtxt(slice_lines)
        Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
        Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
        transpose = np.transpose(Htmp, (0, 3, 2, 1))
        Htmp = transpose
        if np.size(H) == 0:
            H = Htmp
        else:
            H = np.concatenate((H, Htmp), axis=0)
    return H


def npy_file_name_converter(Ridx, Fileidx, CAT):
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]
    return PathRaw + '/' + Prefix + CAT + str(Fileidx) + '.npy'


def tensor_file_name_converter(Ridx, Fileidx):
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]
    return PathRaw + '/' + Prefix + 'NET' + str(Fileidx) + str(time.time()) + '.pth'


def read_npy_of_file(Ridx, Fileidx, CAT):
    filename = npy_file_name_converter(Ridx, Fileidx, CAT)
    if os.path.isfile(filename):
        data = np.load(filename)
        print("Loading " + filename)
        return data
    else:
        print(filename + " does not exist")
        return None


def read_all_of_huaweicup(Ridx, Fileidx):
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]
    # 查找文件夹中包含的所有比赛/测试数据文件，非本轮次数据请不要放在目标文件夹中
    files = os.listdir(PathRaw)
    names = []

    for f in sorted(files):
        if f.find('CfgData') != -1 and f.endswith('.txt'):
            names.append(f.split('CfgData')[-1].split('.txt')[0])

    ## 创建对象并处理
    ## 这里我们读取 输入1
    Didx = str(Fileidx)
    print('Processing Round ' + str(Ridx) + ' Case ' + Didx)

    # 读取配置文件 RoundYCfgDataX.txt
    print('Loading configuration data file')
    cfg_path = PathRaw + '/' + Prefix + 'CfgData' + Didx + '.txt'
    bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num = read_cfg_file(cfg_path)

    # 读取锚点位置文件 RoundYInputPosX.txt
    print('Loading input position file')
    anch_pos_path = PathRaw + '/' + Prefix + 'InputPos' + Didx + '.txt'
    anch_pos = read_anch_file(anch_pos_path, anch_samp_num)

    csi_file_npy = PathRaw + '/' + Prefix + 'InputData' + Didx + '.npy'
    csi_file_txt = PathRaw + '/' + Prefix + 'InputData' + Didx + '.txt'
    if os.path.isfile(csi_file_npy):
        # 读取信道文件 RoundYInputDataX.txt，这里已经读取到了.npy中，直接加载即可
        print('Loading input CSI data of ' + 'Case ' + Didx)
        H = np.load(csi_file_npy)  # 后续可以直接load数据文件
        print("Loading Channel CSI succeed")
    else:
        print("Channel CSI not exist. Reading data from origin txt")
        H = read_channel_csi(csi_file_txt, tol_samp_num, sc_num, ant_num, port_num)
        print("Loading Channel CSI from txt succeed", H[:5])
        # 保存 CSI 为 npy
        np.save(csi_file_npy, H)
        print('Channel CSI saved: ' + csi_file_npy)
    del H  # 释放中间结果内存

    # 文件路径和文件名设置
    geo_file = PathRaw + '/' + Prefix + 'GEO' + Didx + '.npy'
    if os.path.isfile(geo_file):
        # 如果文件存在，则加载数据
        d_geo = np.load(geo_file)
        print("Loading GEO data succeed")
    else:
        d_geo = None
        print("Channel GEO data not exist")

    return bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num, anch_pos, H, d_geo
