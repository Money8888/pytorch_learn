'''
AlexNet
1、5层卷积和2层全连接隐藏层，以及1个全连接输出层
2、AlexNet将sigmoid激活函数改成了更加简单的ReLU激活函数，ReLU激活函数在正区间的梯度恒为1
3、AlexNet通过丢弃法来控制(全连接层)的模型复杂度。而LeNet并没有使用丢弃法
4、AlexNet引入了大量的图像增广，如翻转、裁剪和颜色变化，从而进一步扩大数据集来缓解过拟合
'''
import torch
import torchvision
import torch.utils.data as Data
import sys
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 卷积层
        self.conv = torch.nn.Sequential(
            # 输入通道，输出通道，核大小，步长
            torch.nn.Conv2d(1, 96, 11, 4),
            # 激活函数
            torch.nn.ReLU(),
            # 池化层
            torch.nn.MaxPool2d(3, 2),

            # 减小卷积窗口，使用(填充)为2来使得输入与输出的高和宽一致，且增大输出通道数
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),

            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )

        # 连接层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256 * 5 * 5, 4096),
            torch.nn.ReLU(),
            # 丢弃
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10
            torch.nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


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
    num_epochs = 1
    batch_size = 128
    # 加载数据
    train_batch, test_batch = loadData(batch_size, resize=224)
    model = AlexNet()
    print(model)
    # 学习率
    lr = 0.001
    # 优化器选择Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(model, train_batch, test_batch, batch_size, optimizer, device, num_epochs)

    pre_acc = predict(model, test_batch)

    print("预测精准度为", pre_acc)
