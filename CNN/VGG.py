'''
连续使用数个相同的填充为1、窗口形状为3×3的卷积层后接上一个步幅为2、窗口形状为2×2的最大池化层。
卷积层保持输入的高和宽不变，而池化层则对其减半
'''
import time
import torch
import torch.utils.data as Data
import torchvision
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        # x shape: (batch_size, *, *, ...)
        return x.view(x.shape[0], -1)

def vgg_block(convs_num, in_channels, out_channels):
    '''
    :param convs_num: 卷积层个数
    :param in_channels: 输入的通道数
    :param out_channels: 输出的通道数
    :return: 
    '''
    block = []
    for i in range(convs_num):
        if i == 0:
            # 处理首个卷积层
            block.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            block.append(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        #激活函数
        block.append(torch.nn.ReLU())
    # 池化层
    block.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
    return torch.nn.Sequential(*block)

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    '''
    :param conv_arch: block超参数元组
    :param fc_features: 进入全连接层的特征数
    :param fc_hidden_units: 隐含层单元数
    :return: 
    '''
    model = torch.nn.Sequential()
    # 卷积层
    for i, (convs_num, in_channels, out_channels) in enumerate(conv_arch):
        model.add_module("vgg_block_" + str(i+1), vgg_block(convs_num, in_channels, out_channels))

    # 全连接层
    model.add_module("fc", torch.nn.Sequential(
        FlattenLayer(),
        torch.nn.Linear(fc_features, fc_hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(fc_hidden_units, fc_hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(fc_hidden_units, 10)
    ))
    return model

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
    batch_size = 64
    # 加载数据
    train_batch, test_batch = loadData(batch_size, resize=224)
    lr = 0.001
    ratio = 8
    conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio),
                       (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio)]
    # 总共5个vgg_block 池化了五次，宽高折半了五次，所以224/2^5 = 7
    # 最后一次卷积的通道数为512
    # c * w * h
    fc_features = 512 * 7 * 7
    # 任意
    fc_hidden_units = 4096
    model = vgg(conv_arch, fc_features // ratio, fc_hidden_units // ratio)
    # 优化器选择Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练
    train(model, train_batch, test_batch, batch_size, optimizer, device, num_epochs)

    pre_acc = predict(model, test_batch)

    print("预测精准度为", pre_acc)