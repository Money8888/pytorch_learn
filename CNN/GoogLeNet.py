'''
并行网络
'''
import time
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import sys

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

class Inception(torch.nn.Module):
    # out_c1~4为四条线路输出层的通道数,多个单元即为数组
    def __init__(self, in_channels, out_c1, out_c2, out_c3, out_c4):
        super(Inception, self).__init__()
        #线路1 单1*1卷积层
        self._route1_1 = torch.nn.Conv2d(in_channels, out_c1, kernel_size=1)
        #线路2 1 x 1卷积层后接3 x 3卷积层
        self._route2_1 = torch.nn.Conv2d(in_channels, out_c2[0], kernel_size=1)
        self._route2_2 = torch.nn.Conv2d(out_c2[0], out_c2[1], kernel_size=3, padding=1)
        #线路3，1 x 1卷积层后接5 x 5卷积层
        self._route3_1 = torch.nn.Conv2d(in_channels, out_c3[0], kernel_size=1)
        self._route3_2 = torch.nn.Conv2d(out_c3[0], out_c3[1], kernel_size=5, padding=2)
        #线路4，3 x 3最大池化层后接1 x 1卷积层
        self._route4_1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self._route4_2 = torch.nn.Conv2d(in_channels, out_c4, kernel_size=1)

    def forward(self, x):
        route1 = F.relu(self._route1_1(x))
        route2 = F.relu(self._route2_2(F.relu(self._route2_1(x))))
        route3 = F.relu(self._route3_2(F.relu(self._route3_1(x))))
        route4 = F.relu(self._route4_2(self._route4_1(x)))
        # 在通道维上连结输出
        return torch.cat((route1, route2, route3, route4), dim=1)

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

def GoogLeNet():
    # 模块1，输出通道为64，卷积层7 * 7，池化层3 * 3
    block1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 模块2 先是64通道的1×1卷积层，然后是将通道增大3倍的3×33×3卷积层
    block2 = torch.nn.Sequential(
        torch.nn.Conv2d(64, 64, kernel_size=1),
        torch.nn.Conv2d(64, 192, kernel_size=3, padding=1),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    block3 = torch.nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    block4 = torch.nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    block5 = torch.nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        GlobalAvgPool2d()
    )

    model = torch.nn.Sequential(
        block1,
        block2,
        block3,
        block4,
        block5,
        FlattenLayer(),
        # 全连接层
        torch.nn.Linear(1024, 10)
    )
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
    batch_size = 128
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_batch, test_batch = loadData(batch_size, resize=96)

    lr, num_epochs = 0.001, 5
    model = GoogLeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, train_batch, test_batch, batch_size, optimizer, device, num_epochs)

    pre_acc = predict(model, test_batch)

    print("预测精准度为", pre_acc)
