import itertools

import numpy as np


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
