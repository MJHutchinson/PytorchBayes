import torch
from torch.nn import Module
import numpy as np

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

        _,_,_,self.y_scale = self.train_loader.get_transforms()
        self.y_scale = torch.Tensor(self.y_scale)
        if cuda:
            self.y_scale = self.y_scale.cuda()


        self.data_size = 0
        self.test_size = 0

        for idx, (inputs, targets) in enumerate(train_loader):
            self.data_size += inputs.size()[0]

        for idx, (inputs, targets) in enumerate(test_loader):
            self.test_size += inputs.size()[0]

    def train_step(self):
        self.model.train()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            targets = targets.unsqueeze(0).repeat(self.train_samples, 1, 1)
            inputs  = inputs.unsqueeze(0).repeat(self.train_samples, 1, 1)

            with torch.enable_grad():
                outputs = self.model.forward_prob(inputs)

                outputs = outputs * self.y_scale
                targets = targets * self.y_scale

                kl = self.model.kl() / self.data_size
                log_lik = self.loss(outputs, targets)
                elbo = - log_lik + kl

                self.optimiser.zero_grad()
                elbo.backward()
                self.optimiser.step()


        return elbo.detach().cpu().numpy().tolist(), log_lik.detach().cpu().numpy().tolist(), kl.detach().cpu().numpy().tolist()


    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        total_log_likelihood = 0.
        total_auxiliary_metric = 0.

        for batch_idx, (inputs, targets) in enumerate(self.test_loader):

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs  = inputs.unsqueeze(0).repeat(self.test_samples, 1, 1)

            outputs = self.model.forward_prob(inputs)

            outputs = outputs * self.y_scale
            targets = targets * self.y_scale

            loss = self.loss.test(outputs, targets)

            # preds = outputs.argmax(dim=1)
            # correct = preds.eq(targets.long()).sum()

            rmse = torch.sqrt((outputs-targets)**2).mean(0).sum()

            total_log_likelihood += loss/self.test_size
            total_auxiliary_metric += rmse/self.test_size

        kl = self.model.kl()

        total_log_likelihood        = total_log_likelihood.cpu().numpy().tolist()
        total_auxiliary_metric      = total_auxiliary_metric.cpu().numpy().tolist()

        return total_log_likelihood, kl / self.data_size, total_auxiliary_metric

    @torch.no_grad()
    def predictions(self, train=False):

        predictions = None
        actuals = None

        if train:
            data = self.train_loader
        else:
            data = self.test_loader

        # Loop over all batches
        for batch_idx, (inputs, targets) in enumerate(data):

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs  = inputs.unsqueeze(0).repeat(self.test_samples, 1, 1)
            preds = self.model.forward_prob(inputs).cpu().numpy()
            targets = targets.cpu().numpy()
            # Compute average loss
            if predictions is None:
                predictions = preds
            else:
                predictions = np.append(predictions, preds, axis=1)

            if actuals is None:
                actuals = targets
            else:
                actuals = np.append(actuals, targets, axis=0)

        return predictions, actuals

    def get_config(self):
        model_config = self.model.config
        optimiser_config = {'lr': self.optimiser.defaults['lr']}
        return {**model_config, **optimiser_config}


    def __str__(self):
        config = self.get_config()
        s = ''
        for key in config:
            s += f'{key}_{config[key]}_'

        s = s[:-1]
        return s
