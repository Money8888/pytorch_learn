'''
多层感知机
含dropout
训练过程中，使用Dropout，其实就是对部分权重和偏置在某次迭代训练过程中，不参与计算和更新而已，
并不是不再使用这些权重和偏置了(预测和预测时，会使用全部的神经元，包括使用训练时丢弃的神经元)
'''
import torch
import numpy as np
from softmax_regress import loadData, get_fashion_mnist_labels, show_fashion_mnist

# 输入层 输出层 隐含层两层节点个数
inputsn, outputsn, hiddensn1, hiddensn2 = 784, 10, 256, 256
# 丢弃法概率
drop_prob1, drop_prob2 = 0.2, 0.5

# 激活函数 选relu函数 relu = max(0,x)
def relu(X):
    return torch.max(X, other=torch.tensor(0.0))

def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()

    return mask * X / keep_prob

def model(X, params, bool_training=True):
    X = X.view((-1, inputsn))
    H1 = relu(torch.matmul(X, params[0]) + params[1])
    if bool_training:
        # 训练模型时才dropout
        # 在第一层全连接后添加丢弃层
        H1 = dropout(H1, drop_prob1)
    H2 = (torch.matmul(H1, params[2]) + params[3]).relu()
    if bool_training:
        # 在第二层全连接后添加丢弃层
        H2 = dropout(H2, drop_prob2)
    return torch.matmul(H2, params[4]) + params[5]

def sgd(params, lr, batch_size):
    for param in params:
        # 注意这里更改param时用的param.data
        param.data -= lr * param.grad / batch_size

def train(train_batch, test_batch, optimizer = None):

    # 初始化参数
    W1 = torch.tensor(np.random.normal(0, 0.01, (inputsn, hiddensn1)), dtype=torch.float)
    b1 = torch.zeros(hiddensn1, dtype=torch.float)
    W2 = torch.tensor(np.random.normal(0, 0.01, (hiddensn1, hiddensn2)), dtype=torch.float)
    b2 = torch.zeros(hiddensn2, dtype=torch.float)
    W3 = torch.tensor(np.random.normal(0, 0.01, (hiddensn2, outputsn)), dtype=torch.float)
    b3 = torch.zeros(outputsn, dtype=torch.float)

    # 参数列表
    params = [W1, b1, W2, b2, W3, b3]
    for param in params:
        param.requires_grad_(requires_grad=True)

    num_epochs, lr = 5, 100.0


    # 交叉熵损失
    loss = torch.nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(params, lr, batch_size)

    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, train_n = 0.0, 0.0, 0
        for trainX, trainy in train_batch:
            # 构造softmax回归
            trainy_pre = model(trainX, params)
            l = loss(trainy_pre, trainy)

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()


            # 梯度后向传播
            l.backward()

            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_loss_sum += l.item()
            train_acc_sum += (trainy_pre.argmax(dim=1) == trainy).float().sum().item()
            train_n += trainy.shape[0]
        train_acc = train_acc_sum / train_n

        # 测试集
        test_acc_sum, test_n = 0, 0
        for testX, testy in test_batch:
            # 测试时不应该使用丢弃法
            if isinstance(model, torch.nn.Module):
                # 针对torch.nn.Module包定义好的模型
                # 评估模式，关闭dropout
                model.eval()
                test_acc_sum += (model(testX, params).argmax(dim=1) == testy).float().sum().item()
                # 改回训练模式
                model.train()
            else:
                # 针对自定义模型
                if("bool_training" in model.__code__.co_varnames):
                    # 如果模型中有bool_training这个参数
                    test_acc_sum += (model(testX, params, bool_training=False).argmax(dim=1) == testy).float().sum().item()
                else:
                    test_acc_sum += (model(testX, params).argmax(dim=1) == testy).float().sum().item()
            test_n += testy.shape[0]
        test_acc = test_acc_sum / test_n

        print('第%d轮的交叉熵损失: %.4f, 训练集acc: %.3f, 测试集acc: %.3f'
          % (epoch + 1, train_loss_sum / train_n, train_acc, test_acc))
    return model, params

def predict(model, params, test_batch):
    predX, predy = iter(test_batch).next()
    true_labels = get_fashion_mnist_labels(predy.numpy())
    pred_labels = get_fashion_mnist_labels(model(predX, params, bool_training=False).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    show_fashion_mnist(predX[0:9], titles[0:9])

if __name__ == "__main__":
    batch_size = 256
    train_batch, test_batch = loadData(batch_size)
    model, params = train(train_batch, test_batch)
    predict(model, params, test_batch)
