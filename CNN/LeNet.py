# 卷积神经网络
import time
import torch
import torch.utils.data as Data
import torchvision
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadData(batch_size):
    # 可下载 训练集，转化为Tensor格式
    mnist_train = torchvision.datasets.FashionMNIST(root='D:/PycharmProjects/pytorch_data/Datasets/FashionMNIST',
                  train=True, download=True, transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='D:/PycharmProjects/pytorch_data/Datasets/FashionMNIST',
                 train=False, download=True, transform=torchvision.transforms.ToTensor())

    # 批量导入
    if sys.platform.startswith("win"):
        works_num = 0
    else:
        works_num = 4
    train_batch = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=works_num)
    test_batch = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=works_num)

    return train_batch, test_batch
'''
卷积层保留输入形状，使图像的像素在高和宽两个方向上的相关性均可能被有效识别
卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算，从而避免参数尺寸过大
'''
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        '''
        在卷积层块中，每个卷积层都使用5×5的窗口，并在输出上使用sigmoid激活函数。
        第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16。
        这是因为第二个卷积层比第一个卷积层的输入的高和宽要小，
        所以增加输出通道使两个卷积层的参数尺寸类似。
        卷积层块的两个最大池化层的窗口形状均为2×2，且步幅为2。
        由于池化窗口与步幅形状相同，池化窗口在输入上每次滑动所覆盖的区域互不重叠
        '''
        # 卷积层
        self.conv = torch.nn.Sequential(
            # 卷积层，输入通道数, 输出通道数, 核大小
            torch.nn.Conv2d(1, 6, 5),
            # 卷积后 为 (1, 6, 24, 24)， 24 = 28 - 5 + 1 即输入层大小-核矩阵大小+1
            # 激活函数
            torch.nn.Sigmoid(),
            # 池化层，池化核方阵大小，步长
            torch.nn.MaxPool2d(2, 2),
            # 池化后 为(1, 6, 12, 12)，因为池化矩阵大小为2 且步长为2 所以就是除以2

            # 增加输出通道使两个卷积层的参数尺寸类似
            torch.nn.Conv2d(6, 16, 5),
            # 卷积后 为(1, 16, 8, 8)，因为8 = 12 - 5 + 1

            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(2, 2)
            # 池化后 为(1,16,4,4)
        )

        # 全连接层
        self.fc = torch.nn.Sequential(
            # 输入样本大小，输出样本大小
            # 通道*高*宽
            torch.nn.Linear(16*4*4, 120),
            torch.nn.Sigmoid(),
            torch.nn.Linear(120, 84),
            torch.nn.Sigmoid(),
            # 10个类别
            torch.nn.Linear(84, 10)
        )

    # 前向传播
    def forward(self, image):
        # 批次*通道*高*宽 = 256, 1, 28, 28
        # print("图像大小为" + str(image.shape))
        feature = self.conv(image)
        output = self.fc(feature.view(image.shape[0], -1))
        return output

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
    batch_size = 256
    # 加载数据
    train_batch, test_batch = loadData(batch_size)
    model = LeNet()
    print(model)
    # 学习率
    lr = 0.001
    # 优化器选择Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(model, train_batch, test_batch, batch_size, optimizer, device, num_epochs)

    pre_acc = predict(model, test_batch)

    print("预测精准度为", pre_acc)

