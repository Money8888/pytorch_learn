import torch
import numpy as np
import matplotlib.pyplot as plt

def template(X):
    noise = np.random.rand(1, 100)
    templates = np.sin(X * np.pi) + noise
    templates = torch.from_numpy(templates).float()
    return templates

def GAN(ideas, components):
    # G层生成层
    G = torch.nn.Sequential(
        torch.nn.Linear(ideas, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, components)
    )
    # D层评价层
    D = torch.nn.Sequential(
        torch.nn.Linear(components, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
        torch.nn.Sigmoid()
    )

    return G, D

def train(X, batch_size, lr_G, lr_D):
    plt.ion()

    D_loss_history = []
    G_loss_history = []

    G, D = GAN(20, 100)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_D)

    for step in range(1000):
        templates = template(X)
        G_ideas = torch.randn(batch_size, 20)
        # 开始生成
        G_template = G(G_ideas)


        p1 = D(templates)
        p2 = D(G_template)

        D_loss = -torch.mean(torch.log(p1) + torch.log(1 - p2))
        G_loss = torch.mean(torch.log(1 - p2))

        D_loss_history.append(D_loss)
        G_loss_history.append(G_loss)

        opt_D.zero_grad()
        # 重用计算图
        D_loss.backward(retain_graph=True)
        opt_D.step()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if step % 50 == 0:
            plt.cla()
            plt.plot(X, G_template.data.numpy(), c='#4AD631')
            plt.plot(X, np.sin(X * np.pi), c='#74BCFF')
            plt.text(-1, 0.75, 'D accuracy=%.2f (0.5 for D to converge)' % p1)
            plt.text(-1, 0.5, 'D score= %.2f (‐1.38 for G to converge)' % -D_loss)
            plt.ylim((-1, 1))
            plt.legend(loc='lower right', fontsize=10)
            plt.draw()

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    X = np.linspace(-5, 5, 100)
    train(X, 100, 0.3, 0.1)