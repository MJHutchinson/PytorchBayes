from .base import MFVIBaseModel
from ..layers.MFVI_linear import MFVILinear
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy


class MFVIClassificationMLP(MFVIBaseModel):

    def __init__(self, input_size, hidden_sizes, output_size, activation_function=nn.ReLU()):
        super().__init__()

        layers = []

        sizes = deepcopy(hidden_sizes)
        sizes.append(output_size)
        sizes.insert(0, input_size)

        for i in range(0, len(sizes)-1):
            layers.append(MFVILinear(sizes[i], sizes[i+1]))
            if i != (len(sizes) - 2):
                layers.append(activation_function)

        self.network = nn.Sequential(*layers)

    def forward(self, input, no_samples):
        input = input.view(input.shape[0], -1)
        input = input[None, :, :]
        input = input.expand(no_samples, -1, -1)
        output = self.network(input)
        return output # F.softmax(output, 2), output

    def log_likelihood(self, predictions, targets):
        return F.cross_entropy(predictions.view(-1, predictions.shape[-1]), targets.flatten(), reduction='sum')

    def auxiliary_metric(self, predictions, targets):
        classes = F.softmax(predictions, dim=1).argmax(dim=1)
        correct = torch.eq(classes, targets)
        return correct.sum()