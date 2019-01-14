from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import torch.nn.functional as F

cuda = torch.cuda.is_available()

class LogHomoskedasticGaussianLoss(_Loss):

    def __init__(self, log_var_init=-6, reduction='mean'):
        super().__init__()
        self.log_var = nn.Parameter(torch.Tensor([log_var_init]))

    def forward(self, inputs, targets):
        log_probs = torch.distributions.Normal(inputs, torch.exp(0.5 * self.log_var)).log_prob(targets)

        if self.reduction is not None:
            log_probs = log_probs.mean() if self.reduction == 'mean' else log_probs.sum()

        return log_probs

    def test(self, inputs, targets):
        log_probs = torch.distributions.Normal(inputs, torch.exp(0.5 * self.log_var)).log_prob(targets)
        correction = torch.log(torch.Tensor([log_probs.size(0)]))
        log_probs = torch.logsumexp(log_probs, 0)
        if cuda:
            correction = correction.cuda()

        log_probs = log_probs - correction

        if self.reduction is not None:
            log_probs = log_probs.mean() if self.reduction == 'mean' else log_probs.sum()

        return log_probs


class CrossEntropyLoss(_Loss):

    def forward(self, inputs, targets):
        n = inputs.size()[-1]
        return F.cross_entropy(inputs.view(-1, n) ,targets.view(-1))

    def test(self, inputs, targets):
        return F.cross_entropy(inputs, targets)