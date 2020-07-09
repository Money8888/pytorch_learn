'''
Sequential 与 ModuleList区别
ModuleList仅仅是一个储存各种模块的列表，
这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），
而且没有实现forward功能需要自己实现
Sequential内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部forward功能已经实现
'''

import torch
from collections import OrderedDict

class Sequential(torch.nn.Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                # add_module方法会将module添加进self._modules(一个OrderedDict)
                self.add_module(key, module)
        else:
            for index, module in enumerate(args):
                self.add_module(str(index), module)

    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input

if __name__ == "__main__":
    modle = Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    )
    print(modle)