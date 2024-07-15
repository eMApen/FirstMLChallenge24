import numpy as np
from sklearn.neighbors import kneighbors_graph

# 生成或读取相异度矩阵 D_pw (这里假设一个随机矩阵作为示例)
L = 100  # 样本数，tol samp num
np.random.seed(0)
D_pw = np.random.rand(L, L)
D_pw = (D_pw + D_pw.T) / 2  # 确保矩阵对称
np.fill_diagonal(D_pw, 0)  # 对角线元素设为0，表示自己到自己的距离

# 超参数 k 的值
k = 20

# 使用 k 近邻图
G_kNN = kneighbors_graph(D_pw, k, mode='distance', include_self=False)

# 将结果转换为稀疏矩阵
G_kNN_matrix = G_kNN.toarray()

# 输出矩阵
print("k近邻图矩阵 G_kNN：")
print(G_kNN_matrix)
