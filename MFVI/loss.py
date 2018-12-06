from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogHomoskedasticGaussianLoss(_Loss):

    def __init__(self, log_var_init=-6):
        super().__init__()
        self.log_var = nn.Parameter(torch.Tensor([log_var_init]))

    def forward(self, inputs, targets):
        return torch.distributions.Normal(inputs, torch.exp(0.5 * self.log_var)).log_prob(targets).mean()

    def test(self, inputs, targets):
        inputs = inputs.mean(dim=0)
        return torch.distributions.Normal(inputs, torch.exp(0.5 * self.log_var)).log_prob(targets).mean()


class CrossEntropyLoss(_Loss):

    def forward(self, inputs, targets):
        n = inputs.size()[-1]
        return F.cross_entropy(inputs.view(-1, n) ,targets.view(-1))

    def test(self, inputs, targets):
        return F.cross_entropy(inputs, targets)