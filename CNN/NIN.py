'''
串联多个由卷积层和“全连接”层构成的小网络来构建一个深层网络
全连接层用1 * 1的卷积层代替
'''
import torch
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
import sys
import time

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

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    '''
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: 核矩阵大小
    :param stride: 步长
    :param padding: 填充大小
    :return: 
    '''
    block = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        torch.nn.ReLU(),
        # 全连接层，1*1卷积层
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=1),
        torch.nn.ReLU()
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


def NIN():
    '''
    0 output shape:  torch.Size([1, 96, 54, 54])
    1 output shape:  torch.Size([1, 96, 26, 26])
    2 output shape:  torch.Size([1, 256, 26, 26])
    3 output shape:  torch.Size([1, 256, 12, 12])
    4 output shape:  torch.Size([1, 384, 12, 12])
    5 output shape:  torch.Size([1, 384, 5, 5])
    6 output shape:  torch.Size([1, 384, 5, 5])
    7 output shape:  torch.Size([1, 10, 5, 5])
    8 output shape:  torch.Size([1, 10, 1, 1])
    9 output shape:  torch.Size([1, 10])
    '''
    modle = torch.nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        torch.nn.Dropout(0.5),

        # 类别
        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        GlobalAvgPool2d(),
        FlattenLayer(),
    )
    return modle

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

def predict(model, test_batch, device = None):
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
    # X = torch.rand(1, 1, 224, 224)
    # model = NIN()
    # for name, block in model.named_children():
    #     X = block(X)
    #     print(name, 'output shape: ', X.shape)

    num_epochs = 1
    batch_size = 256
    # 加载数据
    train_batch, test_batch = loadData(batch_size, resize=224)
    model = NIN()
    print(model)
    # 学习率
    lr = 0.002
    # 优化器选择Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(model, train_batch, test_batch, batch_size, optimizer, device, num_epochs)

    pre_acc = predict(model, test_batch)

    print("预测精准度为", pre_acc)