# 长记忆神经网络
'''
增加了遗忘门，输入门，输出门
记忆细胞
'''

import torch
import numpy as np
from data_handle import *
import time
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
word_list, word_dict, corpus_index, word_len = data_load()
inputsn, hiddensn, outputsn = word_len, 256, word_len


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((inputsn, hiddensn)),
                _one((hiddensn, hiddensn)),
                torch.nn.Parameter(torch.zeros(hiddensn, device=device, dtype=torch.float32),requires_grad=True))

    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数

    # 输出层参数
    # 输出层参数
    W_hq = _one((hiddensn, outputsn))
    b_q = torch.nn.Parameter(torch.zeros(outputsn, device=device, dtype=torch.float32), requires_grad=True)
    return torch.nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
# lstm单元
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    # 隐含层细胞和记忆细胞
    (H, C) = state
    outputs = []
    for X in inputs:
        # 输出门
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        # 遗忘门
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        # 输出门
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        # 记忆细胞
        C_candidate = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_candidate
        H = O * torch.tanh(C)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)

def sgd(params, lr, batch_size):
    for param in params:
        # 注意这里更改param时用的param.data
        param.data -= lr * param.grad / batch_size

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

# 训练模型
def train_predict(model, init_param, init_rnn_state, hiddensn,
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
                # 为了使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                state.detach_()

            inputs = to_onehot(X, word_len)
            # outputs为num_steps个形状为(batch_size, vocab_size)的矩阵
            outputs, state = model(inputs, state, params)
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
            # perplexity困惑度
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, math.exp(loss_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, model, params, init_rnn_state,
                                        hiddensn, word_len, device, word_list, word_dict))
