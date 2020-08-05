# 门控循环单元
'''
包括更新门和重置门
包含基本实现和简单实现
'''

import torch
import numpy as np
from data_handle import *
import time
import math

from RNN_simple import RNNModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
word_list, word_dict, corpus_index, word_len = data_load()
inputsn, hiddensn, outputsn = word_len, 256, word_len

# 获取初始参数的函数
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((inputsn, hiddensn)),
                _one((hiddensn, hiddensn)),
                torch.nn.Parameter(torch.zeros(hiddensn, device=device, dtype=torch.float32),requires_grad=True))

    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数

    # 输出层参数
    W_hq = _one((hiddensn, outputsn))
    b_q = torch.nn.Parameter(torch.zeros(outputsn, device=device, dtype=torch.float32), requires_grad=True)
    return torch.nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])

def init_gru_state(batch_size, hiddensn, device):
    # 返回初始化的隐藏状态
    return torch.zeros((batch_size, hiddensn), device=device)

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        # 更新门值
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        # 重置门值
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        # 候选状态
        H_candidate = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_candidate
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, H

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
                # 为了使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
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
            # perplexity困惑度
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, math.exp(loss_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                                        hiddensn, word_len, device, word_list, word_dict))

def predict_simple_rnn(prefix, next_num_chars, model, word_len, device, word_list, word_dict):
    state = None
    output = [word_dict[prefix[0]]]

    for t in range(next_num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):
                # LSTM (hidden state和cell state)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(word_dict[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([word_list[i] for i in output])
# 简洁实现
def train_simple_predict(rnn, hiddensn, word_len, device, corpus_index,
                  word_list, word_dict, num_epochs, num_steps, lr,
                  clip_theta, batch_size, pred_period, pred_len, prefixes):
    # 交叉熵损失函数
    loss = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    rnn.to(device)

    state = None

    for epoch in range(num_epochs):
        loss_sum,  n, start =  0.0, 0, time.time()
        # 加载数据集
        data = data_sample_consecutive(corpus_index, batch_size, num_steps, device)

        for X, Y in data:
            if state is not None:
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            (output, state) = rnn(X, state)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)

            losses = loss(output, y.long())

            optimizer.zero_grad()

            losses.backward()

            grad_clip(rnn.parameters(), clip_theta, device)

            optimizer.step()

            loss_sum += losses.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(loss_sum / n)
        except OverflowError:
            perplexity = float('inf')

        if (epoch + 1) % pred_period == 0:
            # perplexity困惑度
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_simple_rnn(prefix, pred_len, rnn, word_len, device, word_list, word_dict))


if __name__ == "__main__":
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']



    # train_predict(gru, get_params, init_gru_state, hiddensn,
    #               word_len, device, corpus_index, word_list,
    #                   word_dict, False, num_epochs, num_steps, lr,
    #                   clipping_theta, batch_size, pred_period, pred_len,
    #                   prefixes)

    ## 简洁实现
    lr = 1e-2
    # 用torch的工具包
    gru_layer = torch.nn.GRU(input_size=word_len, hidden_size=hiddensn)
    model = RNNModel(gru_layer, word_len).to(device)

    train_simple_predict(model, hiddensn, word_len, device, corpus_index, word_list,
                  word_dict, num_epochs, num_steps, lr,
                  clipping_theta, batch_size, pred_period, pred_len,
                  prefixes)

