import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import configurations
import numpy as np

import model
import utility


def get_loaders(file_path: str):
    train_dataset = datasets.MNIST(
        file_path,
        True,
        download=configurations.DOWNLOAD_DATASET,
        transform=configurations.TRANSFORMATION,
    )
    train_dataloader = DataLoader(
        train_dataset,
        configurations.BATCH_SIZE,
        configurations.SHUFFLE,
        num_workers=configurations.NUM_WORKERS,
        pin_memory=configurations.PIN_MEMORY,
        drop_last=configurations.DROP_LAST,
    )
    test_dataset = datasets.MNIST(
        file_path,
        train=False,
        download=configurations.DOWNLOAD_DATASET,
        transform=configurations.TRANSFORMATION,
    )
    test_dataloader = DataLoader(
        test_dataset,
        configurations.BATCH_SIZE,
        configurations.SHUFFLE,
        num_workers=configurations.NUM_WORKERS,
        pin_memory=configurations.PIN_MEMORY,
        drop_last=configurations.DROP_LAST,
    )
    return train_dataloader, test_dataloader


def save_model(network, optim, filename="checkpoint.pth.tar"):
    print("=> saving checkpoint")
    checkpoint = {
        "state_dict": network.state_dict(),
        "optimizer": optim.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_model(filename, network, optim, learningrate):
    print("=> loading checkpoint")
    checkpoint = torch.load(filename, map_location=configurations.DEVICE)
    network.load_state_dict(checkpoint["state_dict"])
    optim.load_state_dict(checkpoint["optimizer"])
    for group in optim.param_groups:
        group["lr"] = learningrate


def calculate_accuracy(
    network: model.MNIST_network, loader: utility.DataLoader, object_tresh=0.5
):
    network.eval()
    num_correct = 0
    num_total = len(loader.dataset)
    # print(num_total)
    for images, targets in tqdm(loader):
        images = images.to(configurations.DEVICE)
        targets = targets.to(configurations.DEVICE)
        with torch.no_grad():
            results = network(images)

        # print(results.shape)  # torch.Size([configurations.BATCH_SIZE, 10])
        results_one = results[..., :] >= object_tresh
        results_zero = results[..., :] < object_tresh
        results[..., :][results_one] = 1
        results[..., :][results_zero] = 0
        # print(results[0])
        # print(targets[0])
        for tensor_index in range(results.shape[0]):
            """print(targets[tensor_index].item())
            target_class = targets[tensor_index].item()"""
            target = torch.functional.F.one_hot(targets[tensor_index], num_classes=10)
            target = target.to(torch.float32)
            # print(target)
            if torch.equal(results[tensor_index], target):
                num_correct += 1
    network.train()
    return round(num_correct / num_total, 3) * 100


def load_image(filename):
    image = torch.load(filename)
    image = np.moveaxis(image[0, :, :, :].detach().numpy(), [0, 1, 2], [2, 0, 1])
    return image
