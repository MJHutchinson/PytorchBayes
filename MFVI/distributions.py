import torch
import numpy as np
log_2pi = np.log(2*np.pi)

class Distribution(object):

    def pdf(self, x):
        raise NotImplementedError

    def logpdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def logcdf(self, x):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Normal(Distribution):

    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar
        self.shape = mu.size()


    def logpdf(self, x):
        return -0.5 * log_2pi \
               - 0.5 * self.logvar \
               - (x - self.mu).pow(2) / (2 * torch.exp(self.logvar))

    def pdf(self, x):
        return torch.exp(self.logpdf(x))

    def sample(self):
        if self.mu.is_cuda:
            eps = torch.cuda.FloatTensor(self.shape).normal_()
        else:
            eps = torch.FloatTensor(self.shape).normal_()
        return self.mu + torch.exp(0.5 * self.logvar) * eps

    def kl(self, distribution):
        if isinstance(distribution, self.__class__):
            const_term = -0.5
            log_var_diff = 0.5 * (-self.logvar + distribution.logvar)
            mu_diff_term = 0.5 * ((self.mu - distribution.mu) ** 2 + torch.exp(self.logvar))/torch.exp(distribution.logvar)

            return const_term + log_var_diff + mu_diff_term
