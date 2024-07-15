import numpy as np


# 核心定位功能函数，由参赛者实现
def calcLoc(H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num):
    '''
    H: 所有点位的信道
    anch_pos:锚点用户ID及坐标
    bs_pos: 基站坐标
    tol_samp_num: 总点位数
    anch_samp_num: 锚点点位数
    port_num: SRS Port数（UE天线数）
    ant_num: 基站天线数
    sc_num: 信道子载波数
    '''

    ######### 以下代码，参赛者用自己代码替代 ################
    ######### 样例代码中直接返回全零数据作为估计结果 ##########

    loc_result = np.zeros([tol_samp_num, 2], 'float')

    return loc_result
