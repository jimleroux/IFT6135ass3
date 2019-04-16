import torch
from torch import nn
import numpy as np
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = "./model/"

def D_KL(mu, log_var):
    var = torch.exp(log_var)
    return 0.5*torch.sum(mu**2 + var - log_var - 1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(3,3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=(5, 5)),
            nn.ELU(),
            nn.Linear(256, 2*100)
        )

    def forward(self, x):
        output = x
        for _, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                output = output.view(output.size()[0], output.size()[1])
            output = layer(output)
        mean, log_var = output[:, :100], output[:, 100:]
        var = torch.exp(log_var)
        e = torch.randn(output.size()[0], output.size()[1]//2).to(DEVICE)
        output = mean + var*e
        return output, mean, log_var

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(100, 256),
            nn.ELU(),
            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4, 4)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=(3,3), padding=(2, 2)),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(2, 2))
        )

    def forward(self, x):
        output = x
        for _, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                output = layer(output)
                output = output.view(output.size()[0], output.size()[1], 1, 1)
            else:
                output = layer(output)
        return output

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()

    def forward(self, x):
        output, mean, log_var = self.encoder(x)
        output = self.decoder(output)
        return self.sigmoid(output), mean, log_var

    def fit(self, trainloader, n_epochs, lr, print_every=1):
        print('Training autoencoder...')
        start_time = time.time()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        for epoch in range(n_epochs):
            self.train()
            train_loss = 0
            for inputs in trainloader:
                inputs = inputs.to(DEVICE)
                recons, mean, log_var = self.forward(inputs)
                recons_loss = -self.bce(recons, inputs)*784
                loss = torch.mean(D_KL(mean, log_var) - recons_loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += loss.data.cpu().numpy() * inputs.shape[0]

            train_loss = train_loss / len(trainloader.dataset)

            if (epoch + 1) % print_every == 0:
                epoch_time = self._get_time(start_time, time.time())
            print('epoch: {} | Train loss: {:.3f} | time: {}'.format(
                    epoch + 1,
                    train_loss,
                    epoch_time)
                )
        self.save(MODEL_DIR)
        print('Autoencoder trained.')

    def _get_time(self, starting_time, current_time):
        total_time = current_time - starting_time
        minutes = round(total_time // 60)
        seconds = round(total_time % 60)
        return '{} min., {} sec.'.format(minutes, seconds)

    def save(self, model_path: str):
        """
        Save model parameters under a generated name
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, model_path+"vae.pt")