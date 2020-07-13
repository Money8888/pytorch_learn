'''
CNN工具函数
'''
import torch

import sys

from CNN.LeNet import loadData, train, predict

sys.path.append("..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cov2d(X, Kernel):
    '''
    :param X: 输入矩阵
    :param Kernel: 核矩阵
    :return: 输出矩阵
    '''
    K_height, K_width = Kernel.shape
    # 输出矩阵Y
    # 维度为(X_height - K_height + 1,X_width - K_width +1)
    Y = torch.zeros((X.shape[0] - K_height + 1, X.shape[1] - K_width + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + K_height, j: j + K_width] * Kernel).sum()
    return Y

##
def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                # 最大值池化
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                # 平均值池化
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

def corr2d_multi_in(X, Kernel):
    '''
    单输出通道函数
    :param X: 含通道的输入
    '''
    res = cov2d(X[0, :, :], Kernel[0, :, :])
    for i in range(1, X.shape[0]):
        res += cov2d(X[i, :, :], Kernel[i, :, :])
    return res

def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])

# 批量归一化
def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    '''
    :param is_training: 是否处于训练
    :param X: 
    :param gamma: 拉伸因子
    :param beta: 偏移量
    :param moving_mean: 移动平均
    :param moving_var: 移动方差
    :param eps: 防止分母为0
    :param momentum: 移动系数
    '''
    if not is_training:
        # 预测模式
        X_pre = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 如果X为(m,n,x,y)则为卷积层，为(x,y)则为全连接层
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_pre = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var
    # 拉伸和偏移
    Y = gamma * X_pre + beta
    return Y, moving_mean, moving_var




class Conv2D(torch.nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return cov2d(x, self.weight) + self.bias


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        # x shape: (batch_size, *, *, ...)
        return x.view(x.shape[0], -1)

# 可以直接用torch.nn.BatchNorm2d
class BatchNorm(torch.nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 初始化拉伸和偏移参数
        self.gamma = torch.nn.Parameter(torch.ones(shape))
        self.beta = torch.nn.Parameter(torch.zeros(shape))

        # 不参与求梯度和迭代的变量
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        # 保存更新过的moving_mean和moving_var, Module实例的training属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = batch_norm(self.training, X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# 通过边缘检测例子来学习核数组
def learn():
    X = torch.ones(6, 8)
    X[:, 2:6] = 0

    Y = torch.zeros(6, 7)
    Y[:, 1] = 1
    Y[:, 5] = -1

    conv2d = Conv2D(kernel_size=(1, 2))

    step = 50
    lr = 0.01

    for i in range(step):
        Y_pre = conv2d.forward(X)
        loss = ((Y_pre - Y) ** 2).sum()
        loss.backward()

        conv2d.weight.data -= lr * conv2d.weight.grad
        conv2d.bias.data -= lr * conv2d.bias.grad

        # 梯度清零
        conv2d.weight.grad.fill_(0)
        conv2d.bias.grad.fill_(0)
        print('Step %d, loss %.3f' % (i + 1, loss.item()))
    print("weight: ", conv2d.weight.data)
    print("bias: ", conv2d.bias.data)



if __name__ == "__main__":
    # 卷积的测试，学习核数组
    # X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    # K = torch.tensor([[0, 1], [2, 3]])
    # print(cov2d(X, K))
    # learn()

    # 批量归一化的测试
    # 生成LeNet网络
    LeNet = torch.nn.Sequential(
        torch.nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
        BatchNorm(6, num_dims=4),
        torch.nn.Sigmoid(),
        torch.nn.MaxPool2d(2, 2),  # kernel_size, stride
        torch.nn.Conv2d(6, 16, 5),
        BatchNorm(16, num_dims=4),
        torch.nn.Sigmoid(),
        torch.nn.MaxPool2d(2, 2),
        FlattenLayer(),
        torch.nn.Linear(16 * 4 * 4, 120),
        BatchNorm(120, num_dims=2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(120, 84),
        BatchNorm(84, num_dims=2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(84, 10)
    )

    batch_size = 256
    train_iter, test_iter = loadData(batch_size=batch_size)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(LeNet.parameters(), lr=lr)
    train(LeNet, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    predict(LeNet, test_iter, device=None)
    print(LeNet[1].gamma.view((-1,)), LeNet[1].beta.view((-1,)))
