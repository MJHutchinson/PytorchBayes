import torch
from torch.nn import Module

cuda = torch.cuda.is_available()

class MeanFieldVariationalInference(Module):

    def __init__(self, model, loss, optimiser, train_loader, test_loader, train_samples=10, test_samples=100):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimiser = optimiser
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_samples = train_samples
        self.test_samples = test_samples

        self.data_size = 0
        self.test_size = 0

        for idx, (inputs, targets) in enumerate(train_loader):
            self.data_size += inputs.size()[0]

        for idx, (inputs, targets) in enumerate(test_loader):
            self.test_size += inputs.size()[0]

    def train_step(self):
        self.model.train()

        reconstruction = 0.
        kl = 0.
        elbo = 0.

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            targets = targets.unsqueeze(0).repeat(self.train_samples, 1)
            inputs  = inputs.view(-1, 28*28).unsqueeze(0).repeat(self.train_samples, 1, 1)

            with torch.enable_grad():
                outputs = self.model.forward_prob(inputs)

                kl = self.model.kl() / self.data_size
                reconstruction = self.loss(outputs, targets)
                elbo = reconstruction + kl

                self.optimiser.zero_grad()
                elbo.backward()
                self.optimiser.step()

        return elbo, reconstruction, kl


    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        total_log_likelihood = 0.
        total_auxiliary_metric = 0.

        for batch_idx, (inputs, targets) in enumerate(self.test_loader):

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs  = inputs.view(-1, 28*28).unsqueeze(0).repeat(self.test_samples, 1, 1)

            outputs = self.model.forward_prob(inputs).mean(dim=0)

            loss = self.loss(outputs, targets)

            preds = outputs.argmax(dim=1)
            correct = preds.eq(targets).sum()

            total_log_likelihood += loss
            total_auxiliary_metric += correct

        kl = self.model.kl()

        total_log_likelihood        = total_log_likelihood.cpu().numpy()
        total_auxiliary_metric      = total_auxiliary_metric.cpu().numpy()

        return total_log_likelihood / self.test_size, kl / self.data_size, total_auxiliary_metric / self.test_size


