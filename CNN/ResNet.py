# 残差网络

import sys
import time
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadData(batch_size, resize=None):

    # 数据增强
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root='D:/PycharmProjects/pytorch_data/Datasets/FashionMNIST',
                                                    train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root='D:/PycharmProjects/pytorch_data/Datasets/FashionMNIST',
                                                   train=False, download=True, transform=transform)

    # 批量导入
    if sys.platform.startswith("win"):
        works_num = 1
    else:
        works_num = 4
    train_batch = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=works_num)
    for X, y in train_batch:
        print(X.shape)
        print(y.shape)
    test_batch = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=works_num)

    return train_batch, test_batch

# 残差层
# 残差块里有2个有相同输出通道数的3×33×3卷积层
class Residual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        '''
        :param in_channels: 输入的通道数
        :param out_channels: 输出的通道数
        :param use_1x1conv: 是否使用1*1卷积层 
        :param stride: 步长
        '''
        super(Residual, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        # 批量归一化
        self.b1 = torch.nn.BatchNorm2d(out_channels)
        self.b2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.b1(self.conv1(X)))
        Y = self.b2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        # 输出和输入，此时Y = F(X) - X为残差层的输出，所以残差层实际拟合的是F(X)-X
        return F.relu(Y + X)

class GlobalAvgPool2d(torch.nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
       return F.avg_pool2d(x, kernel_size=x.size()[2:])

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        # x shape: (batch_size, *, *, ...)
        return x.view(x.shape[0], -1)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    '''
    :param in_channels: 输入层通道数
    :param out_channels: 输出层通道数
    :param num_residuals: 残差层数
    :param first_block: 是否是第一个resnet块
    :return: 
    '''
    if first_block:
        # 第一个块 输入和输出的通道数需一致
        assert in_channels == out_channels
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            block.append(Residual(out_channels, out_channels))
    return torch.nn.Sequential(*block)

def ResNet():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    model.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    model.add_module("resnet_block2", resnet_block(64, 128, 2))
    model.add_module("resnet_block3", resnet_block(128, 256, 2))
    model.add_module("resnet_block4", resnet_block(256, 512, 2))
    model.add_module("global_avg_pool", GlobalAvgPool2d())
    model.add_module("fc", torch.nn.Sequential(FlattenLayer(), torch.nn.Linear(512, 10)))
    return model

def evaluate_accuracy(data_batch, model, device = None):
    if device is None and isinstance(model, torch.nn.Module):
        device = list(model.parameters())[0].device

    acc_sum, n = 0, 0

    # for X, y in train_batch:
    #     print("train_X", X.shape)
    #     print("train_y", y.shape)
    # for X, y in test_batch:
    #     print("test_X", X.shape)
    #     print("test_y", y.shape)

    with torch.no_grad():
        for X, y in data_batch:

            if isinstance(model, torch.nn.Module):
                # 评估模式,关闭dropout
                model.eval()
                acc_sum += (model(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                # 改回训练模式
                model.train()
            else:
                if ('is_training' in model.__code__.co_varnames):
                    # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (model(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (model(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train(model, train_batch, test_batch, batch_size, optimizer, device, num_epochs):
    model = model.to(device)
    print("运行在:" , device)

    # 损失函数,交叉熵函数
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        start = time.time()

        for X, y in train_batch:
            X = X.to(device)
            y = y.to(device)

            # 前向计算
            y_pre = model(X)

            l = loss(y_pre, y)

            # 梯度清零
            optimizer.zero_grad()

            l.backward()
            optimizer.step()

            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_pre.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        test_acc = evaluate_accuracy(test_batch, model)

        print("第%d轮的损失为%.4f,训练acc为%.3f，测试acc %.3f，耗时%.1f sec" %
              (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def predict(model, test_batch, device=None):
    if device is None and isinstance(model, torch.nn.Module):
        device = list(model.parameters())[0].device

    predX, predy = iter(test_batch).next()

    acc_sum, n = 0, 0

    with torch.no_grad():
        if isinstance(model, torch.nn.Module):
            acc_sum += (model(predX.to(device)).argmax(dim=1) == predy.to(device)).float().sum().cpu().item()
        else:
            if ('is_training' in model.__code__.co_varnames):
                # 如果有is_training这个参数
                # 将is_training设置成False

                acc_sum += (model(predX, is_training=False).argmax(dim=1) == predy).float().sum().item()
            else:
                acc_sum += (model(predX).argmax(dim=1) == predy).float().sum().item()
        print("预测值:", model(predX).argmax(dim=1))
    return acc_sum


if __name__ == "__main__":
    # model = ResNet()
    # X = torch.rand((1, 1, 224, 224))
    # for name, layer in model.named_children():
    #     X = layer(X)
    #     print(name, ' output shape:\t', X.shape)
    batch_size = 128
    train_batch, test_batch = loadData(batch_size, resize=96)

    lr, num_epochs = 0.001, 5
    model = ResNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, train_batch, test_batch, batch_size, optimizer, device, num_epochs)

    pre_acc = predict(model, test_batch)

    print("预测精准度为", pre_acc)