import torch


class MeanFieldVariationalInference:
    def __init__(self, network, optimiser):
        self._network = network
        self._optimiser = optimiser

    @torch.no_grad()
    def step(self, data_loader, data_length, no_samples, device):
        self._network.train()

        log_likelihood = 0.
        kl = 0.
        elbo = 0.

        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            target.unsqueeze(0)
            target = target.expand(no_samples, -1)

            with torch.enable_grad():
                output = self._network(data, no_samples)

                log_likelihood = self._network.log_likelihood(output, target)
                kl = self._network.kl()

                reconstruction = log_likelihood / len(data)
                kl = - kl / data_length

                elbo = (reconstruction - kl)

                self._optimiser.zero_grad()
                elbo.backward()
                self._optimiser.step()

        return elbo, log_likelihood, kl

    @torch.no_grad()
    def evaluate(self, data_loader, data_length, no_samples, device):
        self._network.eval()

        total_log_likelihood = 0.
        total_auxiliary_metric = 0.

        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            output = self._network(data, no_samples)
            output = output.mean(dim=0)

            log_likelihood = self._network.log_likelihood(output, target)
            auxiliary_metric = self._network.auxiliary_metric(output, target)

            total_log_likelihood += log_likelihood
            total_auxiliary_metric += auxiliary_metric

        kl = self._network.kl()

        total_log_likelihood        = total_log_likelihood.cpu().numpy()
        total_auxiliary_metric      = total_auxiliary_metric.cpu().numpy()
        kl                          = kl.cpu().numpy()

        return total_log_likelihood / data_length, kl / data_length, total_auxiliary_metric / data_length


