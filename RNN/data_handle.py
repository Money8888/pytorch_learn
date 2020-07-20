# 数据预处理
import zipfile
import random
import torch



def data_load():
    with zipfile.ZipFile(r"D:\PycharmProjects\pytorch_learn\RNN\data\jaychou_lyrics.txt.zip") as zin:
        with zin.open("jaychou_lyrics.txt") as f:
            corpus_chars = f.read().decode('utf-8')
    # 换行符替换成空格,并只取前10000个字
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]

    # 建立字典索引
    word_list = list(set(corpus_chars))
    word_dict = dict([char, i] for i, char in enumerate(word_list))
    corpus_index = [word_dict[char] for char in corpus_chars]
    word_len = len(word_dict)
    return word_list, word_dict, corpus_index, word_len


def data_sample_random(corpus_index, batch_size, num_steps, device=None):
    '''
    随机采样,每个样本是原始序列上任意截取的一段序列。相邻的两个随机小批量在原始序列上的位置不一定相毗邻
    无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态,在训练模型时，每次随机采样前都需要重新初始化隐藏状态
    :param corpus_index: 索引值
    :param batch_size: 
    :param num_steps: 每个样本所包含的时间步数
    '''
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_index) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_index = list(range(num_examples))
    random.shuffle(example_index)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_index[pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成每轮训练的数据集
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * epoch_size
        batch_index = example_index[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_index]
        Y = [_data(j * num_steps + 1) for j in batch_index]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


def data_sample_consecutive(corpus_index, batch_size, num_steps, device=None):
    '''
    相邻采样，这里的相邻指的是一轮中的两个批次的元素是相邻的,而随机采样不一定相邻
    :param corpus_index: 
    :param batch_size: 
    :param num_steps: 
    :param device: 
    :return: 
    '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_index = torch.tensor(corpus_index, dtype=torch.float32, device=device)
    data_len = len(corpus_index)
    batch_len = data_len // batch_size
    index = corpus_index[0: batch_size * batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = index[:, i: i + num_steps]
        Y = index[:, i + 1: i + num_steps + 1]
        yield X, Y


if __name__ == "__main__":
    my_seq = list(range(30))
    for X, Y in data_sample_random(my_seq, batch_size=2, num_steps=6):
        print('X: ', X, '\nY:', Y, '\n')
    # dataHandle()