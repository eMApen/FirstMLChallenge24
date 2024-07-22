import logging

import networkx as nx
import numpy as np
from tqdm import tqdm

# 尝试导入 cugraph 和 cudf，如果失败则使用 networkx
try:
    import cudf
    import cugraph

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def construct_knn_graph(D_pw, k):
    """构建 k-近邻图的邻接矩阵"""
    # 将对角线元素置为0
    np.fill_diagonal(D_pw, 0)
    n = D_pw.shape[0]
    knn_graph = np.zeros_like(D_pw)
    for i in range(n):
        # 排除自身，即对 D_pw[i] 的元素进行排序后取前 k 个，但不包括自身
        knn_idx = np.argsort(D_pw[i])
        knn_idx = knn_idx[knn_idx != i][:k]
        for j in knn_idx:
            knn_graph[i, j] = D_pw[i, j]
            knn_graph[j, i] = D_pw[j, i]  # 无向图
    logging.info(
        f'构建的 k-近邻图，这里对k进行排序，把最小的k个邻接矩阵搬到图里，并将对角线置为0：\n{knn_graph[:10, :10]}')  # 打印前 10 行 10 列用于检查
    return knn_graph


def compute_shortest_paths_dijkstra(knn_graph):
    """使用 NetworkX 和 Dijkstra 算法计算最短路径长度矩阵"""
    G = nx.from_numpy_array(knn_graph, create_using=nx.Graph)
    num_nodes = knn_graph.shape[0]
    shortest_paths = np.zeros((num_nodes, num_nodes))

    logging.info(f'开始计算最短路径...')
    for i in tqdm(range(num_nodes), desc="计算最短路径节点数", leave=True):
        lengths = nx.single_source_dijkstra_path_length(G, i)
        for j in lengths:
            shortest_paths[i, j] = lengths[j]

    logging.info(f'计算完成，最短路径矩阵的前 10 行 10 列：\n{shortest_paths[:10, :10]}')
    return shortest_paths


def compute_shortest_paths_dijkstra_cugraph(knn_graph):
    """使用 cuGraph 和 Dijkstra 算法计算最短路径长度矩阵"""
    # 将邻接矩阵转换为 COO 格式
    sources, targets = knn_graph.nonzero()
    weights = knn_graph[sources, targets]
    df = cudf.DataFrame({
        'src': sources,
        'dst': targets,
        'weight': weights
    })

    # 创建 cuGraph 图
    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weight', renumber=True)

    num_nodes = knn_graph.shape[0]
    shortest_paths = np.full((num_nodes, num_nodes), np.inf)  # 初始化为无穷大

    logging.info(f'开始计算最短路径...')
    for i in tqdm(range(num_nodes), desc="计算最短路径节点数", leave=True):
        lengths = cugraph.shortest_path(G, i)
        for j in range(num_nodes):
            if j in lengths['vertex'].to_numpy():
                shortest_paths[i, j] = lengths[lengths['vertex'] == j]['distance'].values[0]

    logging.info(f'计算完成，最短路径矩阵的前 10 行 10 列：\n{shortest_paths[:10, :10]}')
    return shortest_paths


if __name__ == '__main__':

    size = 20000
    k = 20

    # 示例相异度矩阵
    A = np.random.rand(size, size)
    D_pw = (A + A.T) / 2
    np.fill_diagonal(D_pw, 0)

    np.set_printoptions(precision=5, threshold=10, edgeitems=10, linewidth=200)
    logging.info(f'构建的相异度矩阵：\n{D_pw[:10, :10]}')  # 打印前 10 行 10 列用于检查

    # 构建 k-近邻图
    logging.info('开始构建 k-近邻图...')
    knn_graph_matrix = construct_knn_graph(D_pw, k)

    # 选择使用 CuGraph 或 NetworkX 计算最短路径长度矩阵 D_geo
    if GPU_AVAILABLE:
        logging.info('检测到 GPU，使用 CuGraph 计算最短路径...')
        D_geo = compute_shortest_paths_dijkstra_cugraph(knn_graph_matrix)
        # logging.info('再用 NetworkX 计算最短路径...')
        # D_geo_net = compute_shortest_paths_dijkstra(knn_graph_matrix)
    else:
        logging.info('未检测到 GPU，使用 NetworkX 计算最短路径...')
        D_geo = compute_shortest_paths_dijkstra(knn_graph_matrix)

    # 保存 D_geo
    # np.save('D_geo.npy', D_geo)
    logging.info('最短路径矩阵已保存。')
