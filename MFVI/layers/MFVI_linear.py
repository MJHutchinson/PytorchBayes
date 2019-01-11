import torch
import torch.nn as nn
from .MFVI_module import MFVIModule

cuda = torch.cuda.is_available()

class MFVILinear(MFVIModule):

    def __init__(self, input_size, output_size, prior_mean=0, prior_var=1):
        super(MFVILinear, self).__init__()

        self.input_size  = input_size
        self.output_size = output_size

        self.W_m = nn.Parameter(torch.Tensor(input_size, output_size))
        self.W_v = nn.Parameter(torch.Tensor(input_size, output_size))
        self.b_m = nn.Parameter(torch.Tensor(output_size))
        self.b_v = nn.Parameter(torch.Tensor(output_size))

        self.W_m.data.normal_(mean=0, std=0.1)
        self.W_v.data.fill_(-11.0)
        self.b_m.data.normal_(mean=0, std=0.1)
        self.b_v.data.fill_(-11.0)

        self.prior_W_mean = nn.Parameter(torch.Tensor([prior_mean]), requires_grad=False)
        self.prior_W_var  = nn.Parameter(torch.Tensor([prior_var]), requires_grad=False)
        self.prior_b_mean = nn.Parameter(torch.Tensor([prior_mean]), requires_grad=False)
        self.prior_b_var  = nn.Parameter(torch.Tensor([prior_var]), requires_grad=False)

    def forward(self, act, use_reparametisation=True):

        K = act.shape[0]
        N = act.shape[1]

        if use_reparametisation:
            m_a   = torch.einsum('kni,io->kno', (act, self.W_m))
            v_a   = torch.einsum('kni,io->kno', (act**2, torch.exp(self.W_v)))

            eps_a = torch.normal(mean=torch.zeros(K, N, self.output_size)).to(self.W_m.device)
            eps_b = torch.normal(mean=torch.zeros(K, 1, self.output_size)).to(self.W_m.device)

            a = m_a + eps_a * torch.sqrt(v_a + 1e-9)
            b = self.b_m + eps_b * torch.exp(0.5 * self.b_v)
            return a + b

    def kl(self):
        m, v = self.W_m, self. W_v
        m0, v0 = self.prior_W_mean, self.prior_W_var

        const_term = -0.5 * self.input_size * self.output_size
        log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
        mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m) ** 2) / v0)

        W_KL = const_term + log_std_diff + mu_diff_term

        m, v = self.b_m, self.b_v
        m0, v0 = self.prior_b_mean, self.prior_b_var

        const_term = -0.5 * self.input_size * self.output_size
        log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
        mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m) ** 2) / v0)

        b_KL = const_term + log_std_diff + mu_diff_term

        KL = W_KL + b_KL

        return KL