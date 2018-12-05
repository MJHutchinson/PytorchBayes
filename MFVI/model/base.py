from MFVI.layers.MFVI_module import MFVIModule


class MFVIBaseModel(MFVIModule):

    def __init__(self):
        super(MFVIBaseModel, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def log_likelihood(self, predictions, targets):
        raise NotImplementedError

    def auxiliary_metric(self, predictions, targets):
        raise NotImplementedError

    def train_step(self, data_loader, no_samples, data_length):
        self.train()

        total_elbo = 0.
        total_log_likelihood = 0.

        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimiser.zero_grad()
            output = self(data, no_samples)

            target.unsqueeze(0)
            target = target.expand(no_samples, -1)

            log_likelihood = self.log_likelihood(output, target)
            kl = self.kl()

            elbo = log_likelihood/len(data) - kl/data_length

            elbo.backward()
            self.optimiser.step()

            total_elbo += elbo
            total_log_likelihood += log_likelihood

        return total_elbo / data_length, total_log_likelihood / data_length, self.kl() / data_length

    def evaluate(self, data_loader, no_samples, data_length, device):
        self.eval()

        total_log_likelihood = 0.
        total_auxiliary_metric = 0.

        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output = self(data, no_samples)
            output = output.mean(dim=0)

            log_likelihood = self.log_likelihood(output, target)
            auxiliary_metric = self.auxiliary_metric(output, target)

            total_log_likelihood += log_likelihood
            total_auxiliary_metric += auxiliary_metric

        kl = self.kl()

        return total_log_likelihood / data_length, kl / data_length, total_auxiliary_metric / data_length


