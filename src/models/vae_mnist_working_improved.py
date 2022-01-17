"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
from torch.optim import Adam
from numpy.core.multiarray import asarray
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

import wandb
import argparse
parser = argparse.ArgumentParser()
args, leftovers = parser.parse_known_args()
config_defaults = {
    "batch_size": 100,
    "x_dim": 784,
    "hidden_dim": 400,
    "latent_dim": 20,
    "learning_rate": 1e-3,
    "epochs": 10,
}

if hasattr(args, 'epochs'):
    config_defaults["epochs"] = args.epochs
if hasattr(args, 'learning_rate'):
    config_defaults["learning_rate"] = args.learning_rate

wandb.init(config=config_defaults)

config = wandb.config

# Model Hyperparameters
dataset_path = 'datasets/'
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = config.batch_size
x_dim = config.x_dim
hidden_dim = config.hidden_dim
latent_dim = config.latent_dim
lr = config.learning_rate
epochs = config.epochs

if False:
    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(
        dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(
        dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1, shuffle=True)
    train_images = []
    train_labels = []
    for idx, (x, y) in enumerate(train_loader):
        train_images.append(x)
        train_labels.append(y)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1, shuffle=False)
    test_images = []
    test_labels = []
    for idx, (x, y) in enumerate(test_loader):
        test_images.append(x)
        test_labels.append(y)

    m1 = {'images': train_images, 'labels': train_labels}
    torch.save(m1, dataset_path+"MNIST/processed/train.pt")
    m2 = {'images': test_images, 'labels': test_labels}
    torch.save(m2, dataset_path+"MNIST/processed/test.pt")

dataset_path = dataset_path + '/MNIST/Processed/'


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, file_name, path=dataset_path, verbose=False):
        data = torch.load(path+file_name)
        self.images = data['images']
        self.labels = data['labels']
        #self.images = self.images.astype(np.float)
        #self.labels = self.labels.astype(np.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # return torch.tensor(self.images[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
        return self.images[idx], self.labels[idx]


train_dataset = CustomDataSet('train.pt')
test_dataset = CustomDataSet('test.pt')

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.training = True

    def forward(self, x):
        h_ = torch.relu(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        std = torch.exp(0.5*log_var)
        z = self.reparameterization(mean, std)

        return z, mean, log_var

    def reparameterization(self, mean, std,):
        epsilon = torch.rand_like(std)

        z = mean + std*epsilon

        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.relu(self.FC_hidden(x))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, x):
        z, mean, log_var = self.Encoder(x)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim,
                  latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim,
                  hidden_dim=hidden_dim, output_dim=x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

wandb.watch(model, log_freq=100)


BCE_loss = nn.BCELoss()


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(
        x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)

print("Start training VAE...")
model.train()
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)

        overall_loss += loss.item()
        loss.backward()
        optimizer.step()
    # Validation loss
    validation_loss = 0
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)

        validation_loss += loss.item()
    wandb.log({"validation_loss": validation_loss / (batch_idx*batch_size)})
    wandb.log({"loss": overall_loss / (batch_idx*batch_size)})
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ",
          overall_loss / (batch_idx*batch_size))
print("Finish!!")

# Generate reconstructions
model.eval()
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)
        x_hat, _, _ = model(x)
        break

save_image(x.view(batch_size, 1, 28, 28), 'orig_data.png')
save_image(x_hat.view(batch_size, 1, 28, 28), 'reconstructions.png')

# Generate samples
with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    generated_images = decoder(noise)

save_image(generated_images.view(
    batch_size, 1, 28, 28), 'generated_sample.png')

wandb.log({
    'original': wandb.Image(x.view(batch_size, 1, 28, 28)),
    'reconstructed': wandb.Image(x_hat.view(batch_size, 1, 28, 28)),
    'generated': wandb.Image(generated_images.view(batch_size, 1, 28, 28))
})
