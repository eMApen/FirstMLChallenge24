import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def plt_point_hull(point):
    # 取出x和y轴的数据
    x = point[:, 1]
    y = point[:, 2]

    # 绘制散点图和膨胀后的凸包
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=10, c='blue', alpha=0.5)

    hull = ConvexHull(point)
    # 绘制膨胀后的凸包边界
    for simplex in hull.simplices:
        plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')

    plt.title('2D Scatter Plot with Expanded Convex Hull')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.show()
