# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def make_data(load_path,store_path):
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(
        load_path, transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(
        load_path, transform=mnist_transform, train=False, download=True)

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
    torch.save(m1, store_path+"/train.pt")
    m2 = {'images': test_images, 'labels': test_labels}
    torch.save(m2, store_path+"/test.pt")

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    make_data(input_filepath,output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
