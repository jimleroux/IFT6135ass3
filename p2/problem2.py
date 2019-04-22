import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
from scipy.special import logsumexp
import math

from dataloader import TrainDataset, ValidDataset, TestDataset
from VAE import VAE, reconstruction_loss

MODEL_PATH = "./model/vae.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def gaussian_density(x, mean, sigma):
    num = torch.exp(-(x - mean)**2/(2*sigma**2))
    dem = (math.sqrt(2*math.pi))*(sigma)
    return num/dem

def get_dataloader(batch_size=256):
    trainset = TrainDataset()
    validset = ValidDataset()
    testset = TestDataset()
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, validloader, testloader

def get_model():
    vae = VAE(DEVICE)
    vae.load(MODEL_PATH)
    vae = vae.to(DEVICE)

    return vae

def evaluate(vae, trainloader, validloader, testloader):
    elbo_train = vae.evaluate(trainloader)
    elbo_valid = vae.evaluate(validloader)
    elbo_test = vae.evaluate(testloader)

    print("Evaluating the model...")
    print("Train ELBO: {} | Valid ELBO: {} | Test ELBO: {}".format(elbo_train, elbo_valid, elbo_test))

def importance_sampling(VAE, inputs, K=200):
    sum_arg = []
    VAE.eval()
    with torch.no_grad():
        inputs = inputs.to(DEVICE)
        _, mean, log_sigma = VAE.encoder.forward(inputs)
        sigma = torch.exp(log_sigma)
        z = torch.randn((inputs.shape[0], K, mean.shape[1])).to(DEVICE) #shape(M, K, L)
        for samp in range(K):
            z_i = mean + sigma*z[:, samp, :]
            p_vals = gaussian_density(z_i, 0, 1)
            q_vals = gaussian_density(z_i, mean, sigma)
            recons = VAE.decoder.forward(z_i)
            log_p_z = torch.sum(torch.log(p_vals), dim=1)
            log_q_z = torch.sum(torch.log(q_vals), dim=1)
            log_p_xz = reconstruction_loss(recons, inputs).squeeze(1)
            sumarg = log_p_xz + log_p_z - log_q_z
            sumarg = sumarg.cpu().numpy()
            sum_arg.append(sumarg)

        return  -np.log(K) + logsumexp(sum_arg, axis=0)

def estimate_loglh(dataloader, VAE, K=200):
    log_lh = 0.
    for inputs in dataloader:
        log_pi = importance_sampling(VAE, inputs, K)
        log_lh += np.sum(log_pi)
    log_lh = log_lh/len(dataloader.dataset)
    return log_lh

def main(args):
    trainloader, validloader, testloader = get_dataloader(args.batch_size)
    vae = get_model()

    if args.evaluate:
        evaluate(vae, trainloader, validloader, testloader)

    if args.estimate:
        print("Estimating log_lh...")
        log_lh_valid = estimate_loglh(validloader, vae, args.K)
        log_lh_test = estimate_loglh(testloader, vae, args.K)
        print("Valid Log_Likelihood: {} | Test Log_Likelihood: {}".format(log_lh_valid, log_lh_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        type=int,
                        default=100)
    parser.add_argument('--K',
                        type=int,
                        default=200,
                        help="Number of sampling.")
    parser.add_argument('--evaluate',
                        action="store_true",
                        help="Evaluate model.")
    parser.add_argument('--estimate',
                        action="store_true",
                        help="Estimate log_lh.")                        
    args = parser.parse_args()
    main(args)