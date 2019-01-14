import torch
import torch.nn as nn
from torch.nn import Module
from copy import deepcopy
import numpy as np

cuda = torch.cuda.is_available()


class MLP(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, activation_function=nn.ReLU()):
        super().__init__()

        self.config = {
            'hidden_size': hidden_sizes,
        }

        layers = []

        sizes = deepcopy(hidden_sizes)

        sizes.append(output_size)
        sizes.insert(0, input_size)

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i != (len(sizes) - 2):
                layers.append(activation_function)

        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return self.layers(input)


class MLPTrainer(Module):

    def forward(self, *input):
        pass

    def __init__(self, model, loss, optimiser, train_loader, test_loader):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimiser = optimiser
        self.train_loader = train_loader
        self.test_loader = test_loader

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

            with torch.enable_grad():
                outputs = self.model(inputs)

                outputs = outputs * self.y_scale
                targets = targets * self.y_scale

                loss = self.loss(outputs, targets).sum()

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

        return loss.detach().cpu().numpy().tolist()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        total_loss = 0.

        for batch_idx, (inputs, targets) in enumerate(self.test_loader):

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = self.model(inputs)

            outputs = outputs * self.y_scale
            targets = targets * self.y_scale

            loss = self.loss(outputs, targets).sum()

            total_loss += loss / self.test_size

        total_loss = total_loss.cpu().numpy().tolist()

        return total_loss

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
                inputs = inputs.cuda()

            preds = self.model(inputs).cpu().numpy()
            # Compute average loss
            if predictions is None:
                predictions = preds
            else:
                predictions = np.append(predictions, preds, axis=0)

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
