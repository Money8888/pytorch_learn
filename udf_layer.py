
import torch
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__()
    def forward(self, x):
        return x - x.mean()

class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        # 加入参数
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

if __name__ == "__main__":
    layer = CenteredLayer()
    layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
    model = Dense()
    print(model)