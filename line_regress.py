'''
pytorch实现线性回归

batch_size 设置的稍微小一些比较好，不容易欠拟合
'''
import torch
import numpy as np
import random
from IPython import display
from matplotlib import pyplot as plt

# 生成数据集
def loadData():
    w = [[1],[2],[3]]
    b = 3.5
    x = torch.randn(1000,len(w), dtype=torch.float32)
    y = x.mm(torch.FloatTensor(w)) + b
    y += torch.tensor(np.random.normal(0, 0.001, size=y.size()), dtype=torch.float32)
    return x, y

# 每次返回batch_size（批量大小）个随机样本的特征和标签
def readData(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 样本的读取顺序是随机的
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 最后一次可能不足一个batch
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

def train():
    # 载入数据
    X, y = loadData()

    set_figsize()
    plt.scatter(X[:, 1].numpy(), y.numpy(), 1)
    # 初始化w，b
    w = torch.tensor(np.random.normal(0, 0.001, (X.size()[1], 1)), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)

    # 允许进行梯度下降计算
    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)

    lr = 0.03
    num_epochs = 3
    net = model
    loss = loss_func
    batch_size = 10

    for epoch in range(num_epochs):
        # 训练模型一共需要num_epochs个迭代周期
        for X, y in readData(batch_size, X, y):
            # 计算损失
            l = loss_func(net(X, w, b), y).sum()
            # 梯度后向传播
            l.backward()
            # 更新模型参数
            SGD([w, b], lr, batch_size)

            ## 梯度清零
            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(X, w, b), y)
        print("第%d轮训练的误差为%f" % (epoch + 1, train_l.mean().item()))

    print(w, '\n')
    print(b, '\n')
    plt.show()

# 定义模型
def model(X, w, b):
    return X.mm(torch.FloatTensor(w)) + b

# 定义损失函数
# 平方损失函数
def loss_func(y_pre, y):
    return (y_pre - y.view(y_pre.size())) ** 2

# 定义优化算法
# 随机梯度下降
def SGD(params, lr, batch_size):
    '''
    :param params: 模型参数
    :param lr: 学习率
    :param batch_size: 批次大小 
    '''
    for param in params:
        # 更改param
        param.data -= lr * param.grad / batch_size

def use_svg_display():
    # 矢量图表示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize



if __name__ == "__main__":
    train()