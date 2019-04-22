import torch
from torch import nn
import numpy as np
import time
import scipy.stats

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = "./model/"

def reconstruction_loss(recons, inputs):
    bce = nn.BCELoss(reduction="none")
    recons_loss = -bce(recons, inputs)
    recons_loss = torch.sum(recons_loss, dim=(2, 3))
    return recons_loss

def D_KL(mu, log_sigma):
    """
    Compute KL_div based on the hypothesis that the prior of z is N(0,1).
    """
    sigma = torch.exp(log_sigma)
    return 0.5*torch.sum(mu**2 + sigma**2 - torch.log(sigma**2) - 1, dim=1)

def VAE_loss(recons, inputs, mean, log_sigma):
    """
    Return the ELBO. We train the network with this loss.
    The reconstruction loss is the binary cross entropy loss.
    """
    recons_loss = reconstruction_loss(recons, inputs)
    loss = -torch.mean(recons_loss - D_KL(mean, log_sigma))
    return loss

class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.device = device

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
        mean, log_sigma = output[:, :100], output[:, 100:]
        sigma = torch.exp(log_sigma)
        e = torch.randn(output.size()[0], output.size()[1]//2).to(self.device)
        output = mean + sigma*e
        return output, mean, log_sigma

class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        self.sigmoid = nn.Sigmoid()

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
        return self.sigmoid(output)

class VAE(nn.Module):
    def __init__(self, device=DEVICE):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = Encoder(device)
        self.decoder = Decoder(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)

    def forward(self, x):
        output, mean, log_sigma = self.encoder(x)
        output = self.decoder(output)
        return output, mean, log_sigma

    def fit(self, trainloader, n_epochs, lr, print_every=1):
        print('Training autoencoder...')
        start_time = time.time()
        for epoch in range(n_epochs):
            self.train()
            train_loss = 0
            for inputs in trainloader:
                inputs = inputs.to(self.device)
                recons, mean, log_sigma = self.forward(inputs)
                loss = VAE_loss(recons, inputs, mean, log_sigma)
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
        print("Saving model...")
        self.save(MODEL_DIR)
        print('Autoencoder trained.')

    def evaluate(self, dataloader):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(self.device)
                recons, mean, log_sigma = self.forward(inputs) 
                loss = -VAE_loss(recons, inputs, mean, log_sigma)
                total_loss += loss.data.cpu().numpy() * inputs.shape[0]
        
        return total_loss/len(dataloader.dataset)

    def _get_time(self, starting_time, current_time):
        total_time = current_time - starting_time
        minutes = round(total_time // 60)
        seconds = round(total_time % 60)
        return '{} min., {} sec.'.format(minutes, seconds)

    def save(self, model_path: str):
        """
        Save model parameters
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, model_path+"vae.pt")

    def load(self, model_path: str):
        """
        Restore the model parameters
        """
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])