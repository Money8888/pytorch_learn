'''
多项式拟合
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

train_num, test_num = 100, 20

def set_figsize(figsize=(3.5, 2.5)):
    display.set_matplotlib_formats('svg')
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)

def loadData():
    # 拟合y=1.2x−3.4x2+5.6x3+5+ϵ
    w, b = [1.2, -3.4, 5.6], 5
    features = torch.randn((train_num + test_num, 1))
    # 展开成len(features)行矩阵，列分别对应一次，二次，三次
    poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
    labels = (w[0] * poly_features[:,0] + w[1] * poly_features[:,1] + w[2] * poly_features[:,2] + b)
    # 加上扰动
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

    return features, poly_features, labels

num_epochs, loss = 100, torch.nn.MSELoss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    print(train_features.shape[-1])
    print("+++++++++++++++++++++++++++++++++++++++")
    net = torch.nn.Linear(train_features.shape[-1], 1)
    # 通过Linear文档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化了

    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # sgd下降
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)
    plt.show()

if __name__ == "__main__":
    features, poly_features, labels = loadData()
    # 多项式拟合
    # fit_and_plot(poly_features[:train_num, :], poly_features[train_num:, :],
    #              labels[:train_num], labels[train_num:])

    # 线性拟合
    fit_and_plot(features[:train_num, :], features[train_num:, :],
                 labels[:train_num], labels[train_num:])