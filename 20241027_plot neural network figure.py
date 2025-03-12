# 在代码中导入networkx 和 matplotlib
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()  # 使用DiGraph创建一个有向图G

# 网络包括了5个节点
# 第一层的节点编号为1、2，第2层的是3、4、5
G.add_edge(1, 3)  # 从1到3
G.add_edge(1, 4)  # 从1到4
G.add_edge(1, 5)  # 从1到5
G.add_edge(2, 3)  # 从2到3
G.add_edge(2, 4)  # 从2到4
G.add_edge(2, 5)  # 从2到5

# 创建字典pos，字典的key是节点的名称 
# 字典的value，是节点所在位置

# 1号和2号节点在一列
# 3、4、5在一列
# 因此设置1和2的x坐标为0；3、4、5的为1

# 同一组中的节点，可以均匀的分布在同一列上
# 所以我们将1和2的y坐标，设置为0.25和0.75
# 3、4、5的y坐标0.2、0.5、0.8

# {节点名称:(节点x坐标,节点y坐标)}
pos = {
    1: (0, 0.25),  # 节点1的坐标(0,0.25)
    2: (0, 0.75),  # 节点2的坐标(0,0.75)
    3: (1, 0.2),  # 节点3的坐标(1, 0.2)
    4: (1, 0.5),  # 节点4的坐标(1, 0.5)
    5: (1, 0.8),  # 节点5的坐标(1, 0.8)
}

# 使用nx.draw函数进行绘制
nx.draw(G,  # 要绘制的图
        pos,  # 图中节点的坐标
        with_labels=True,  # 绘制节点的名称
        node_color='white',  # 节点的颜色
        edgecolors='black',  # 边的颜色
        linewidths=3,  # 节点的粗细
        width=2,  # 边的粗细
        node_size=1000  # 节点的大小
        )

plt.show()


# 根据传入的输入层、隐含层、输出层的神经元数量，绘制对应的神经网络
def draw_network_digraph(input_num, hidden_num, output_num):
    G = nx.DiGraph()  # 创建一个图G

    # 连接输入层和隐含层之间的边
    for i in range(input_num):
        for j in range(hidden_num):
            G.add_edge(i, input_num + j)

    # 连接隐含层和输出层之间的边
    for i in range(hidden_num):
        for j in range(output_num):
            G.add_edge(input_num + i, input_num + hidden_num + j)

    pos = dict()  # 计算每个节点的坐标pos
    # 节点的坐标，(x,y)设置为：
    # (0,i-input_num/2)
    # (1,i-hidden_num)/2)
    # (2,i-output_num/2)
    # 根据每一层的节点数量，将节点从中间，向两边分布
    for i in range(0, input_num):
        pos[i] = (0, i - input_num / 2)
    for i in range(0, hidden_num):
        hidden = i + input_num
        pos[hidden] = (1, i - hidden_num / 2)
    for i in range(0, output_num):
        output = i + input_num + hidden_num
        pos[output] = (2, i - output_num / 2)

    # 调用 nx.draw 绘制神经网络
    nx.draw(G,  # 要绘制的图
            pos,  # 图中节点的坐标
            with_labels=False,  # 绘制节点的名称
            node_color='white',  # 节点的颜色
            edgecolors='black',  # 边的颜色
            linewidths=3,  # 节点的粗细
            width=2,  # 边的粗细
            node_size=1000  # 节点的大小
            )


if __name__ == '__main__':
    #   尝试多组参数，绘制不同结构的神经网络
    draw_network_digraph(4, 30, 2)
    plt.show()
    draw_network_digraph(5, 2, 6)
    plt.show()
    draw_network_digraph(1, 10, 1)
    plt.show()
