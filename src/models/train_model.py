from torch.optim import Adam
import torch
import torch.nn as nn
from torchvision.utils import save_image

from vae import *
from data_loader import TrainLoader,TestLoader

import wandb
import argparse

def train_and_test():
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

    wandb.init(config=config_defaults,project="MNIST VAE SWEEP")

    config = wandb.config

    # Model Hyperparameters
    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if cuda else "cpu")
    batch_size = config.batch_size
    x_dim = config.x_dim
    hidden_dim = config.hidden_dim
    latent_dim = config.latent_dim
    lr = config.learning_rate
    epochs = config.epochs

    # importing data loader
    train_loader = TrainLoader(batch_size)
    test_loader = TestLoader(batch_size)

    # Defining model
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim,
                    latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim,
                    hidden_dim=hidden_dim, output_dim=x_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

    # Defining loss and omptimizer
    BCE_loss = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Initializing logging
    wandb.watch(model, log_freq=100)

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

    filename = 'trained_model.pt'
    torch.save(model, './models/'+filename)

if __name__ == '__main__':
    train_and_test()
