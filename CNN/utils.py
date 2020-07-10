'''
CNN工具函数
'''
import torch


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
        res += cov2d(X[i, :, :], K[i, :, :])
    return res

def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])


class Conv2D(torch.nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return cov2d(x, self.weight) + self.bias



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
    # X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    # K = torch.tensor([[0, 1], [2, 3]])
    # print(cov2d(X, K))
    learn()