import time

import math
import torch
import numpy as np
from data_handle import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
word_list, word_dict, corpus_index, word_len = data_load()
inputsn, hiddensn, outputsn = word_len, 256, word_len

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

class RNNModel(torch.nn.Module):
    def __init__(self, rnn_layer, word_len):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.word_len = word_len
        self.dense = torch.nn.Linear(self.hidden_size, word_len)
        self.state = None

    def forward(self, inputs, state):
        # 转化为one-hot
        X = to_onehot(inputs, self.word_len)
        Y, self.state = self.rnn(torch.stack(X), state)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

def predict_rnn(prefix, next_num_chars, model, word_len, device, word_list, word_dict):
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

# 开始训练
def train_predict(rnn,  hiddensn, word_len, device, corpus_index,
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
                print(' -', predict_rnn(prefix, pred_len, rnn, word_len, device, word_list, word_dict))


if __name__ == "__main__":
    rnn_layer = torch.nn.RNN(input_size=inputsn, hidden_size=hiddensn)
    rnn = RNNModel(rnn_layer, word_len).to(device)
    num_epochs, batch_size, lr, clipping_theta, num_steps  = 250, 32, 1e-3, 1e-2, 35
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    train_predict(rnn, hiddensn, word_len, device, corpus_index, word_list,
                  word_dict, num_epochs, num_steps, lr,
                  clipping_theta, batch_size, pred_period, pred_len,
                  prefixes)






