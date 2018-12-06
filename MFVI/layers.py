import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .distributions import *
from numbers import Number

cuda = torch.cuda.is_available()

class Module_MFVI(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError()

    def forward_prob(self, input):
        raise NotImplementedError()

    def kl(self):
        raise NotImplementedError()


class Linear_MFVI(Module_MFVI):

    def __init__(self, input_features, output_features, p_mu=0, p_logvar=0):

        super().__init__()

        self.input_features = input_features
        self.output_features = output_features

        if isinstance(p_mu, Number):
            p_mu = torch.Tensor([p_mu])

        if isinstance(p_logvar, Number):
            p_logvar = torch.Tensor([p_logvar])

        self.p_mu     = Parameter(torch.Tensor(p_mu)    , requires_grad=False)
        self.p_logvar = Parameter(torch.Tensor(p_logvar), requires_grad=False)

        self.p = Normal(self.p_mu, self.p_logvar)

        self.qw_mean   = Parameter(torch.Tensor(input_features, output_features))
        self.qw_logvar = Parameter(torch.Tensor(input_features, output_features))

        self.qw = Normal(self.qw_mean, self.qw_logvar)

        self.qb_mean = Parameter(torch.Tensor(output_features))
        self.qb_logvar = Parameter(torch.Tensor(output_features))

        self.qb = Normal(self.qw_mean, self.qw_logvar)

        self.reset_parameters()


    def reset_parameters(self):
        self.qw_mean.data.normal_(std=0.1)
        self.qb_mean.data.normal_(std=0.1)
        self.qw_logvar.data.fill_(-6)
        self.qb_logvar.data.fill_(-6)


    def forward(self, *input):
        raise NotImplementedError

    def forward_prob(self, input):
        out_mean  = torch.einsum("kni,io->kno", (input, self.qw_mean)) + self.qb_mean # F.linear(input,        weight=self.qw_mean,              bias=self.qb_mean)
        out_var   = torch.einsum("kni,io->kno", (input.pow(2), torch.exp(self.qw_logvar))) + torch.exp(self.qb_logvar) # F.linear(input.pow(2), weight=torch.exp(self.qw_logvar), bias=torch.exp(self.qb_logvar))
        out_sigma = torch.sqrt(1e-8 + out_var)

        if cuda:
            eps = torch.randn(out_mean.size()).cuda()
        else:
            eps = torch.randn(out_mean.size())

        return out_mean + eps * out_sigma

    def kl(self):
        return torch.sum(self.qw.kl(self.p)) + torch.sum(self.qb.kl(self.p))