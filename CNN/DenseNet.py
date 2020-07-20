# 稠密连接网络，区别于ResNet，将输出与后面的层相连而不是和输入相加
import time

import sys
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
        works_num = 0
    else:
        works_num = 4
    train_batch = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=works_num)
    test_batch = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=works_num)

    return train_batch, test_batch

# 卷积块
def conv_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )
    return block

# 过渡块
def transition_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1),
        torch.nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return block

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


class DenseBlock(torch.nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        '''
        :param num_convs: 卷积层个数
        '''
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_channels_temp =  in_channels + i * out_channels
            net.append(conv_block(in_channels_temp, out_channels))
        self.net = torch.nn.ModuleList(net)
        # 计算输出通道
        self.out_channels = in_channels + num_convs * out_channels

    def forward(self, X):
        for block in self.net:
            Y = block(X)
            # 在通道维上将输入和输出连结
            X = torch.cat((X, Y), dim=1)
        return X


def build():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    # 添加稠密层
    # num_channels为当前的通道数,growth_rate增长的通道数
    num_channels, growth_rate = 64, 32
    # 4块，每层卷积层个数
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        db = DenseBlock(num_convs, num_channels, growth_rate)
        model.add_module("DenseBlosk_%d" % i, db)
        # 上一个稠密块的输出通道数
        num_channels = db.out_channels
        # 在稠密块之间加入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            # 排除最后一层
            model.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    model.add_module("BN", torch.nn.BatchNorm2d(num_channels))
    model.add_module("relu", torch.nn.ReLU())
    # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
    model.add_module("global_avg_pool", GlobalAvgPool2d())
    model.add_module("fc", torch.nn.Sequential(FlattenLayer(), torch.nn.Linear(num_channels, 10)))
    return model

def evaluate_accuracy(data_batch, model, device = None):
    if device is None and isinstance(model, torch.nn.Module):
        device = list(model.parameters())[0].device

    acc_sum, n = 0, 0

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
    # X = torch.rand((1, 1, 96, 96))
    # for name, layer in model.named_children():
    #     X = layer(X)
    #     print(name, ' output shape:\t', X.shape)
    batch_size = 128
    train_batch, test_batch = loadData(batch_size, resize=96)
    lr, num_epochs = 0.001, 5
    model = build()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, train_batch, test_batch, batch_size, optimizer, device, num_epochs)

    pre_acc = predict(model, test_batch)

    print("预测精准度为", pre_acc)