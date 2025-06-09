import torch
import torch.nn as nn


class MNIST_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, targets):
        return self.loss(output, targets)


if __name__ == "__main__":
    result_dummy = torch.randn(4, 10)
    target_dummy = torch.randn(4, 10)
    loss_function = MNIST_loss()
    loss = loss_function(result_dummy, target_dummy)
    print(loss)
