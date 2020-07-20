# 循环神经网络
import time

import math
import torch
import numpy as np
from data_handle import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

word_list, word_dict, corpus_index, word_len = data_load()
inputsn, hiddensn, outputsn = word_len, 256, word_len


# one-hot编码
def one_hot(x, n_class, dtype = torch.float32):
    '''
    如果一个字符的索引是整数i, 则创建一个全0的长为N的向量，并将其位置为i的元素设成1
    :param x: 
    :param n_class: 不同字符的数量,等于字典大小
    :param dtype: 
    :return: 
    '''
    x = x.long()
    result = torch.zeros(x.shape[0], n_class, dtype=dtype, device=device)
    result.scatter_(1, x.view(-1, 1), 1)
    return result

def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

# 初始化模型参数
def init_param():
    def _norm(shape):
        tt = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(tt, requires_grad=True)

    # 隐含层参数
    W_xh = _norm((inputsn, hiddensn))
    W_hh = _norm((hiddensn, hiddensn))
    b_h = torch.nn.Parameter(torch.zeros(hiddensn, device=device), requires_grad=True)
    # 输出层参数
    W_hq = _norm((hiddensn, outputsn))
    b_q = torch.nn.Parameter(torch.zeros(outputsn,device=device), requires_grad=True)
    return torch.nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])

def init_rnn_state(batch_size, hiddensn, device):
    # 返回初始化的隐藏状态
    return torch.zeros((batch_size, hiddensn), device=device)

def grad_clip(params, theta, device):
    '''
    :param theta: 梯度的阈值
    '''
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

def sgd(params, lr, batch_size):
    for param in params:
        # 注意这里更改param时用的param.data
        param.data -= lr * param.grad / batch_size

def rnn(inputs, state, params):
    # inputs和outputs维度皆为(batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []

    for X in inputs:
        # 进入隐含层计算
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, H

def predict_rnn(prefix, next_num_chars, rnn, params, init_rnn_state,
                hiddensn, word_len, device, word_list, word_dict):
    '''
    :param prefix: 单词前缀
    :param next_num_chars: 预测的下一个单词的长度
    '''
    # 初始化第一个隐含层状态
    state = init_rnn_state(1, hiddensn, device)
    output = [word_dict[prefix[0]]]

    for t in range(next_num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), word_len)
        # 计算输出和更新隐藏状态
        Y, state = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(word_dict[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([word_list[i] for i in output])


def train_predict(rnn, init_param, init_rnn_state, hiddensn,
                          word_len, device, corpus_index, word_list,
                          word_dict, is_random_random, num_epochs, num_steps,
                          lr, clip_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_random:
        # 是否随机采样
        data_fn = data_sample_random
    else:
        data_fn = data_sample_consecutive

    params = init_param()
    # 损失函数为交叉熵
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_random:
            # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, hiddensn, device)
        else:
            state = None
        loss_sum, n, start = 0.0, 0, time.time()
        data = data_fn(corpus_index, batch_size, num_steps, device)

        for X, Y in data:
            if is_random_random:
                # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, hiddensn, device)
            else:
                state.detach_()

            inputs = to_onehot(X, word_len)
            # outputs为num_steps个形状为(batch_size, vocab_size)的矩阵
            outputs, state = rnn(inputs, state, params)
            # 进行拼接，拼接后为batch_size * num_steps的向量
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)

            # 使用交叉熵损失计算平均分类误差
            losses = loss(outputs, y.long())

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            # 损失反馈
            losses.backward()

            # 裁剪梯度
            grad_clip(params, clip_theta, device)

            # sgd随机梯度下降
            sgd(params, lr, 1)

            loss_sum += losses.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, math.exp(loss_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                                        hiddensn, word_len, device, word_list, word_dict))

if __name__ == "__main__":
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    train_predict(rnn, init_param, init_rnn_state, hiddensn,
                          word_len, device, corpus_index, word_list,
                          word_dict, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)