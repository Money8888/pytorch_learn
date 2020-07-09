## 说明
- 根据《动手学深度学习》自学项目代码，纯手敲
- 项目代码经过重构整合
- 部分代码会有公式的推导
- 加油

# pytorch 学习
#### 如何对内存中的数据进行操作？
torch.Tensor是存储和变换数据的主要工具，提供GPU计算和自动求梯度等更多功能
### pytorch 基础
##### 创建
操作 | 含义
---|---
torch.empty(m,n) | 创建m行n列的未初始化的空张量数组，值为0
torch.rand(m,n) | 创建m行n列的随机张量数组
torch.zeros(m,n, dtype=torch.long) | mxn的long型全0的tensor数组
torch.tensor([5.5, 3]) | 根据列表创建
x.new_ones((m,n,dtype=torch.float64)) | 可以不用声明x,直接构建mxn的全1数组
x.randn_like(x, dtype=torch.float) | 通过现有的Tensor来创建
size()或shape() | 获取维度

##### 操作

操作 | 代码
---|---
加等算术 | x+y 或 torch.add(x, y)或y.add_(x)
索引 | index_select(input, dim, index)
改变形状 | view() 或 reshape()， x.clone().view(15)
转化 | item() 转化为标量数字

> 索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改

> view仅仅是改变了对这个张量的观察角度，内部数据并未改变

> 具体操作参看 [官方文档](https://pytorch.org/docs/stable/tensors.html)

##### 广播机制
不同维度的数组进行操作时，会触发广播机制，将其转化为维度相同
##### 内存开销
y[:] = y + x 不会开辟新内存而y = y+x会开辟
##### 和numpy互转
// 不开辟新内存
torch.from_numpy(a)和a.numpy()
// 开辟新内存
torch.tensor(a)

### softmax 回归
#### 交叉熵
>
```math
\sum^{j \to q} -y_j * log(\hat{y})
```

###  感知机
```math
H = \phi(XW_h + b_h)

O = HW_o + b_o
```
```math
\phi 为激活函数,包括ReLU函数、sigmoid函数和tanh函数
```
> L2正则化 也可以理解为权重衰减，因为对正则项求导后再梯度下降导致w变成原来的(1 - a*λ)倍

> optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd) # 对权重参数衰减

> optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  # 不对偏差参数衰减

### 丢弃法dropout
设丢弃概率为p，那么有p的概率h_i会被清零,h_i为第i个隐含层神经元的激活函数输出
```math
\grave{h_i} = \frac{\xi}{1 - p} h_i
```
期望
```math
E(\grave{h_i}) = \frac{E(\xi)}{1 - p} h_i = h_i
```
丢弃法不改变其输入的期望值
```
def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()

    return mask * X / keep_prob
```
> 只在训练模型时使用丢弃法

### 读写与存储
使用save函数和load函数分别存储和读取Tensor
- save
```
# 保存模型
torch.save(model.state_dict(), PATH)
```
save使用Python的pickle实用程序将对象进行序列化，然后将序列化的对象保存到disk，使用save可以保存各种对象,包括模型、张量和字典等
- load
```
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
```
使用pickle unpickle工具将pickle的对象文件反序列化为内存
