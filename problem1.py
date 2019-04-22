import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from samplers import distribution1, distribution2, distribution3, distribution4


def js_obj(dx, dy):
    js = torch.log(torch.FloatTensor([2]))\
        + 0.5*(torch.mean(torch.log(dx+0.00001))+torch.mean(torch.log(1 - dy+0.00001)))
    return js

def wasserstein_obj(tx, ty, gtz, lam=10):
    wd = torch.mean(tx) - torch.mean(ty)\
        - lam * torch.mean((torch.sqrt(torch.sum(gtz**2, dim=1))-1)**2)
    return wd

def discriminator_obj(dx, dy):
    return torch.mean(torch.log(dx+0.000001))+torch.mean(torch.log(1 - dy+0.000001))


class MLP_Disc(nn.Module):
    def __init__(self, dim):
        super(MLP_Disc, self).__init__()
        self.dim = dim
        self.denses = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),                      
            nn.Linear(256, self.dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.denses(x)
        return out
    
    def _train(self, dist1, dist2, obj, lr=0.001, epochs=500):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_train = []
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            x = torch.FloatTensor(next(dist1))
            y = torch.FloatTensor(next(dist2))            
            outx = self(x)
            outy = self(y)
            criterion = -obj(outx, outy)
            criterion.backward()
            optimizer.step()
            # if epoch % 50 == 0:
            #     print(criterion)
            loss_train.append(criterion)
        return self.jensen_shannon(dist1, dist2)

    def jensen_shannon(self, dist1, dist2):
        x = torch.FloatTensor(next(dist1))
        y = torch.FloatTensor(next(dist2))
        js = (torch.mean(torch.log(2*self(x)+0.00001)) + torch.mean(torch.log(2*(1-self(y))+0.00001))) / 2
        return js

        
class MLP_WD(nn.Module):
    def __init__(self, dim):
        super(MLP_WD, self).__init__()
        self.dim = dim
        self.denses = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),                        
            nn.Linear(256, self.dim)
        )
    
    def forward(self, x):
        out = self.denses(x)
        return out
    
    def _train(self, dist1, dist2, epochs=100):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        uniform = iter(distribution2(512))
        loss_train = []
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            x = torch.FloatTensor(next(dist1))
            y = torch.FloatTensor(next(dist2))    
            alpha = torch.FloatTensor(next(uniform)[:,[0]])
            z = alpha*(torch.FloatTensor(next(dist1))) + (1 - alpha)*torch.FloatTensor(next(dist2))
            z.requires_grad = True
            z.retain_grad()
            outx = self(x)
            outy = self(y)
            outz = self(z)
            torch.sum(outz).backward()
            criterion = -wasserstein_obj(outx, outy, z.grad)
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
            loss_train.append(criterion)
        return self.wasserstein_dist(dist1, dist2)

    def wasserstein_dist(self, dist1, dist2):
        x = torch.FloatTensor(next(dist1))
        y = torch.FloatTensor(next(dist2))
        return torch.mean(self(x)) - torch.mean(self(y))


def problem1_3():
    phis = np.arange(-1, 1.1, 0.1)
    wds = []
    jss = []
    for phi in phis:
        p = distribution1(0, 512)
        q = distribution1(phi, 512)
        func1 = MLP_WD(dim=2)
        func2 = MLP_Disc(dim=2)
        wds.append(func1._train(p, q, epochs=200))
        jss.append(func2._train(p, q, js_obj, epochs=100))
    plt.figure()
    plt.plot(phis, torch.FloatTensor(jss).detach().numpy(), "-sk")
    plt.xlabel(r"$\phi$")
    plt.ylabel("Jensen-Shanon estimate")
    plt.figure()
    plt.plot(phis, torch.FloatTensor(wds).detach().numpy(), "-sk")
    plt.xlabel(r"$\phi$")
    plt.ylabel("Wasserstein distance estimate")
    plt.show()
    return wds, jss

if __name__ == "__main__":
    wds, jss = problem1_3()
    