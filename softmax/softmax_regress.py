import torch
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import sys

# 图片参数
inputn = 784
outputn = 10

# 导入数据集
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

def softmax(X):
    expX = X.exp()
    all_expX = expX.sum(dim = 1, keepdim=True)
    return expX / all_expX

def sgd(params, lr, batch_size):
    for param in params:
        # 注意这里更改param时用的param.data
        param.data -= lr * param.grad / batch_size


def train(train_batch, test_batch, num_epochs, lr=None, optimizer=None):
    '''
    :param train_batch: 
    :param test_batch: 
    :param num_epochs: 
    :param lr: 
    :param optimizer: 
    :return: 返回模型参数w，b
    '''

    # 载入数据

    # 初始化参数
    # softmax回归的权重和偏差参数分别为784×10和1×101×10的矩阵

    # 初始化权值
    w = torch.tensor(np.random.normal(0, 0.01, (inputn, outputn)), dtype=torch.float)
    b = torch.zeros(outputn, dtype=torch.float)
    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)



    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, train_n = 0.0, 0.0, 0
        for trainX, trainy in train_batch:
            # 构造softmax回归
            trainy_pre = softmax(torch.mm(trainX.view((-1, inputn)), w) + b)

            # 损失函数，交叉熵损失
            # gather(input, dim, index)
            # 根据index，在dim维度上选取数据，输出的size与index一样
            loss = (- torch.log(trainy_pre.gather(1, trainy.view(-1, 1)))).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            # 梯度后向传播
            loss.backward()

            if optimizer is None:
                sgd([w, b], lr, batch_size)
            else:
                optimizer.step()

            train_loss_sum += loss.item()
            train_acc_sum += (trainy_pre.argmax(dim=1) == trainy).float().sum().item()
            train_n += trainy.shape[0]
        train_acc = train_acc_sum / train_n

        # 测试集
        test_acc_sum, test_n = 0, 0
        for testX, testy in test_batch:
            testy_yre = softmax(torch.mm(testX.view((-1, inputn)), w) + b)
            test_acc_sum += (testy_yre.argmax(dim=1) == testy).float().sum().item()
            test_n += testy.shape[0]
        test_acc = test_acc_sum / test_n

        print('第%d轮的交叉熵损失: %.4f, 训练集acc: %.3f, 测试集acc: %.3f'
          % (epoch + 1, train_loss_sum / train_n, train_acc, test_acc))
    return w, b


# 预测
def predict(w, b, test_batch):
    predX, predy = iter(test_batch).next()
    true_labels = get_fashion_mnist_labels(predy.numpy())
    pred_labels = get_fashion_mnist_labels(softmax(torch.mm(predX.view((-1, inputn)), w) + b).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    show_fashion_mnist(predX[0:9], titles[0:9])



# 标签赋值
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def use_svg_display():
    # 矢量图表示
    display.set_matplotlib_formats('svg')

def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    batch_size = 256
    train_batch, test_batch = loadData(batch_size)
    w, b = train(train_batch, test_batch, num_epochs=5, lr=0.1)
    predict(w, b, test_batch)
