'''
torch.utils.data模块提供了有关数据处理的工具
torch.nn模块定义了大量神经网络的层
torch.nn.init模块定义了各种初始化方法
torch.optim模块提供了很多常用的优化算法
torch.nn.MSELoss() MSE损失函数
'''


import torch
import torch.utils.data as Data
import numpy as np
from collections import OrderedDict

class LineNN(torch.nn.Module):
    def __init__(self, n_feature):
        '''
        :param n_feature: X自变量的个数
        '''
        super(LineNN, self).__init__()
        self.linear = torch.nn.Linear(n_feature, 1)

    # 前向传播
    def forward(self, X):
        y = self.linear(X)
        return y


def loadData():
    w = [[1],[2],[3]]
    b = 3.5
    x = torch.randn(10000,len(w), dtype=torch.float32)
    y = x.mm(torch.FloatTensor(w)) + b
    y += torch.tensor(np.random.normal(0, 0.001, size=y.size()), dtype=torch.float32)
    return x, y

# 读取数据
def train():
    X, y = loadData()

    batch_size = 10
    # 使用PyTorch的data包,加载数据
    # 将X与y组合成tensor的dataset格式
    dataset = Data.TensorDataset(X, y)
    readData = Data.DataLoader(dataset, batch_size, shuffle=True)

    # net训练模型
    # 1、第一种
    net = LineNN(X.size()[1])
    # 2、第二种
    netSeq1 = torch.nn.Sequential(torch.nn.Linear(X.size()[1], 1))
    # 3、第三种
    netSeq2 = torch.nn.Sequential()
    netSeq2.add_module('linear', torch.nn.Linear(X.size()[1], 1))
    # netSeq2 add 其他层
    netSeq3 = torch.nn.Sequential(OrderedDict([
        ('linear', torch.nn.Linear(X.size()[1], 1))
        # ......
    ]))

    # 初始化模型参数
    torch.nn.init.normal_(net.linear.weight, mean=0, std=0.01)
    torch.nn.init.constant_(net.linear.bias, val=0)

    # 损失函数
    loss = torch.nn.MSELoss()

    # 优化算法
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in readData:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            # 梯度清零
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print('第 %d轮训练的损失为: %f' % (epoch, l.item()))

    dense = net
    print(dense.linear.weight)
    print(dense.linear.bias)

if __name__ == "__main__":
    train()