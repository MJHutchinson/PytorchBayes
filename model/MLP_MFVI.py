import torch
import torch.nn as nn
import torch.nn.functional as F
import MFVI as mm


class MLP_MFVI(mm.Module_MFVI):

    def __init__(self, input_size, hidden_sizes, output_size, p_mean=0, p_var_log=0, activation_function=nn.ReLU()):
        super().__init__()

        layers = []

        sizes = hidden_sizes

        sizes.append(output_size)
        sizes.insert(0, input_size)

        for i in range(0, len(sizes) - 1):
            layers.append(mm.Linear_MFVI(sizes[i], sizes[i + 1], p_mean, p_var_log))
            if i != (len(sizes) - 2):
                layers.append(activation_function)

        self.layers = nn.ModuleList(layers)

    def forward(self, *input):
        raise NotImplementedError

    def forward_prob(self, input):
        for layer in self.layers:
            if isinstance(layer, mm.Module_MFVI):
                input = layer.forward_prob(input)
            else:
                input = layer.forward(input)

        return input

    def kl(self):
        kl = 0
        for layer in self.layers:
            if isinstance(layer, mm.Module_MFVI):
                kl += layer.kl()

        return kl
