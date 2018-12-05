import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from .MFVI_module import MFVIModule
from functools import reduce
import math


class _MFVIConvNd(MFVIModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, prior_mean=0, prior_var=1):
        super(_MFVIConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.W_m = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.W_v = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.W_m = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.W_v = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.b_m = nn.Parameter(torch.Tensor(out_channels))
            self.b_v = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.prior_W_mean = torch.Tensor([prior_mean]).cuda()
        self.prior_W_var = torch.Tensor([prior_var]).cuda()
        if bias:
            self.prior_b_mean = torch.Tensor([prior_mean]).cuda()
            self.prior_b_var = torch.Tensor([prior_var]).cuda()

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.W_m.data.normal_(mean=0, std=0.1)
        self.W_v.data.fill_(-11.0)
        if self.b_m is not None:
            self.b_m.data.normal_(mean=0, std=0.1)
            self.b_v.data.fill_(-11.0)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.b_m is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def kl(self):
        KL = 0

        m, v = self.W_m, self.W_v
        m0, v0 = self.prior_W_mean, self.prior_W_var

        const_term = -0.5 * reduce(lambda x, y: x*y, self.W_m.shape)
        log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
        mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m) ** 2) / v0)

        KL += const_term + log_std_diff + mu_diff_term

        if self.b_m is not None:
            m, v = self.b_m, self.b_v
            m0, v0 = self.prior_b_mean, self.prior_b_var

            const_term = -0.5 * reduce(lambda x, y: x*y, self.b_m.shape)
            log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
            mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m) ** 2) / v0)

            KL += const_term + log_std_diff + mu_diff_term

        return KL


class MFVIConv2d(_MFVIConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, prior_mean=0, prior_var=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MFVIConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        K = input.shape[0]
        N = input.shape[1]
        H = input.shape[-2]
        W = input.shape[-1]

        eps_w = torch.normal(mean=torch.zeros(K, *self.W_m.shape)).cuda()
        weights = self.W_m + torch.exp(0.5 * self.W_v) * eps_w

        if self.b_m is not None:
            eps_b = torch.normal(mean=torch.zeros(K, *self.b_m.shape)).cuda()
            biases = self.b_m + torch.exp(0.5 * self.b_v) * eps_b

            output = []

            for inpt, weight, bias in zip(torch.unbind(input), torch.unbind(weights), torch.unbind(biases)):
                output.append(F.conv2d(inpt, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups))

            return torch.stack(output)

        else:
            output = []

            for input, weight, bias in zip(torch.unbind(input), torch.unbind(weights)):
                output.append(F.conv2d(input, weight, None, self.stride,
                                       self.padding, self.dilation, self.groups))

            return torch.stack(output)
