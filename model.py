import torch
import torch.nn as nn

# architecture inspired by
# AN ENSEMBLE OF SIMPLE CONVOLUTIONAL NEURAL
# NETWORK MODELS FOR MNIST DIGIT RECOGNITION
# by An et al., 2020
# https://arxiv.org/pdf/2008.10400 (viewed last on 09.06.2025)

architecture = [  # 1x28x28
    (32, 3, 1, 0),  # 32x26x26
    (64, 3, 1, 0),  # 64x24x24
    (128, 3, 1, 0),  # 128x22x22
    (256, 3, 1, 0),  # 256x20x20 (256*20^2 = 102400)
    [102400, 10],
]


class CNN_layer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel, stride, padding)
        self.norm = nn.BatchNorm2d(dim_out)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class FC_layer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.norm = nn.BatchNorm1d(dim_out)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        return self.activation(self.norm(self.linear(x)))


class MNIST_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = self.create_network(architecture)

    def create_network(self, architecture):
        is_cnn_so_far = True
        dim_in = 1
        final_layers = nn.ModuleList()
        for layer in architecture:
            if isinstance(layer, tuple):
                final_layers.append(
                    CNN_layer(dim_in, layer[0], layer[1], layer[2], layer[3])
                )
                dim_in = layer[0]
            elif isinstance(layer, list):
                if is_cnn_so_far:
                    is_cnn_so_far = False

                final_layers.append(FC_layer(layer[0], layer[1]))
        return final_layers

    def forward(self, x):
        for layer in self.network:
            if isinstance(layer, FC_layer):
                x = torch.flatten(x, 1)

            x = layer(x)
        return x


if __name__ == "__main__":
    test_tensor = torch.randn(4, 1, 28, 28)
    network = MNIST_network()
    results = network(test_tensor)
    print(results.shape)
    assert results.shape == (4, 10)
