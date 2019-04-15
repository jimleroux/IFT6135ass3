import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from samplers import distribution2, distribution3


def js_obj(dx, dy):
    js = torch.log(torch.FloatTensor([2]))\
        + 0.5*(torch.mean(torch.log(dx+0.00001))+torch.mean(torch.log(1 - dy+0.00001)))
    return js

def wasserstein_obj(tx, ty, gtz, lam=10):
    wd = torch.mean(ty) - torch.mean(tx)\
        + lam * torch.mean((torch.sqrt(torch.sum(gtz**2, dim=1))-1)**2)
    return wd


class MLP_JS(nn.Module):
    def __init__(self):
        super(MLP_JS, self).__init__()
        self.denses = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),                        
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.denses(x)
        return out
    
    def _train(self, dist1, dist2, epochs=1000):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        loss_train = []
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            x = torch.FloatTensor(next(dist1))+1000
            y = torch.FloatTensor(next(dist2))            
            outx = self(x)
            outy = self(y)
            criterion = -js_obj(outx, outy)
            criterion.backward()
            optimizer.step()
            loss_train.append(criterion)
            print(criterion)
        return self.jensen_shannon(dist1, dist2)

    def jensen_shannon(self, dist1, dist2):
        x = torch.FloatTensor(next(dist1))+1000
        y = torch.FloatTensor(next(dist2))
        js = (torch.mean(torch.log(2*self(x)+0.00001)) + torch.mean(torch.log(2*(1-self(y))+0.00001))) / 2
        return js

        
class MLP_WD(nn.Module):
    def __init__(self):
        super(MLP_WD, self).__init__()
        self.denses = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),                        
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        out = self.denses(x)
        return out
    
    def _train(self, dist1, dist2, epochs=1000):
        optimizer = optim.SGD(self.parameters(), lr=0.001)
        uniform = iter(distribution2(512))
        loss_train = []
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            x = torch.FloatTensor(next(dist1))+1
            y = torch.FloatTensor(next(dist2))    
            alpha = torch.FloatTensor(next(uniform)[:,[0]])
            z = alpha*(torch.FloatTensor(next(dist1)) + 1) + (1 - alpha)*torch.FloatTensor(next(dist2))
            z.requires_grad = True
            z.retain_grad()
            outx = self(x)
            outy = self(y)
            outz = self(z)
            torch.sum(outz, dim=0).backward()
            criterion = wasserstein_obj(outx, outy, z.grad)
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
            loss_train.append(criterion)
            if criterion <= 0.1:
                break
        return self.wasserstein_dist(dist1, dist2)

    def wasserstein_dist(self, dist1, dist2):
        x = torch.FloatTensor(next(dist1))
        y = torch.FloatTensor(next(dist2))
        return torch.mean(self(x)) - torch.mean(self(y))

if __name__ == "__main__":
    dist = distribution3(512)
    x = torch.FloatTensor(next(dist))
    y = torch.FloatTensor(next(dist))
    mlp_js = MLP_JS()
    js = mlp_js._train(dist, dist)
    # b = []
    # for i in range(-100,100):
    #     b.append(mlp(torch.FloatTensor([i/10.])))
    # plt.plot(np.arange(-100,100)/10., b)
    # plt.show()
    mlp_wd = MLP_WD()
    wd = mlp_wd._train(dist, dist, epochs=5000)
