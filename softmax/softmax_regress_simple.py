import sys

import torch

sys.path.append("..")
from collections import OrderedDict
from softmax.softmax_regress import loadData, get_fashion_mnist_labels, show_fashion_mnist

# 图片参数
inputn = 784
outputn = 10


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        # x shape: (batch_size, *, *, ...)
        return x.view(x.shape[0], -1)

def train(train_batch, test_batch):
    # 构建模型
    model = torch.nn.Sequential(
        OrderedDict(
            [
                ('flatten', FlattenLayer()),
                ('linear', torch.nn.Linear(inputn, outputn))
            ]
        )
    )

    # 初始化w,b
    torch.nn.init.normal_(model.linear.weight, mean=0, std=0.01)
    torch.nn.init.constant_(model.linear.bias, val=1)

    # 交叉熵损失函数
    loss = torch.nn.CrossEntropyLoss()

    # sgd优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    num_epochs = 5

    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, train_n = 0.0, 0.0, 0
        for trainX, trainy in train_batch:
            # 构造softmax回归
            trainy_pre = model(trainX)
            l = loss(trainy_pre, trainy)

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()

            # 梯度后向传播
            l.backward()


            optimizer.step()

            train_loss_sum += l.item()
            train_acc_sum += (trainy_pre.argmax(dim=1) == trainy).float().sum().item()
            train_n += trainy.shape[0]
        train_acc = train_acc_sum / train_n

        # 测试集
        test_acc_sum, test_n = 0, 0
        for testX, testy in test_batch:
            testy_yre = model(testX)
            test_acc_sum += (testy_yre.argmax(dim=1) == testy).float().sum().item()
            test_n += testy.shape[0]
        test_acc = test_acc_sum / test_n

        print('第%d轮的交叉熵损失: %.4f, 训练集acc: %.3f, 测试集acc: %.3f'
          % (epoch + 1, train_loss_sum / train_n, train_acc, test_acc))
    return model

def predict(model, test_batch):
    predX, predy = iter(test_batch).next()
    true_labels = get_fashion_mnist_labels(predy.numpy())
    pred_labels = get_fashion_mnist_labels(model(predX).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    show_fashion_mnist(predX[0:9], titles[0:9])

if __name__ == "__main__":
    batch_size = 256
    train_batch, test_batch = loadData(batch_size)
    net = train(train_batch, test_batch)
    predict(net, test_batch)




