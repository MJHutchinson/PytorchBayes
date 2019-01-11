import time
import os
from MFVI.inference import MeanFieldVariationalInference
import torch
import torch.nn as nn
import MFVI as mm
import torch.optim as optim
from torchvision import datasets, transforms
from model.MLP_MFVI import MLP_MFVI
from data.data_sets import RegressionDataloader
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def main():
    # data_set = 'wine-quality-red'
    # data_set = 'concrete'
    data_set = 'protein-tertiary-structure'
    # data_set = 'yacht'
    log_dir = './logs'
    fig_dir = f'{log_dir}/figs'
    writer = SummaryWriter('./logs')

    use_cuda = torch.cuda.is_available()
    batch_size = 900
    test_batch_size = 900
    epochs = 50000

    train_loader = RegressionDataloader(data_set, batch_size, data_dir=os.path.abspath('./data'), train=True, shuffle=True)
    test_loader  = RegressionDataloader(data_set, batch_size, data_dir=os.path.abspath('./data'), train=False, shuffle=False)

    input_size, train_length, output_size = train_loader.get_dims()

    model = MLP_MFVI(input_size, [50, 50], output_size, p_var=1.)
    loss = mm.LogHomoskedasticGaussianLoss()

    if use_cuda:
        model.cuda()
        loss.cuda()

    optimiser = optim.Adam([{'params':model.parameters()}, {'params': loss.parameters()}])

    inference = mm.MeanFieldVariationalInference(model, loss, optimiser, train_loader, test_loader)

    for epoch in range(epochs):
        t = time.time()
        elbo, log_lik, kl = inference.train_step()
        train_time = time.time() - t
        ll, kl, aux = inference.evaluate()
        test_time = time.time() - train_time - t

        writer.add_scalar('metrics/train kl divergence', kl, epoch)
        writer.add_scalar('metrics/train log likelihood', log_lik, epoch)
        writer.add_scalar('metrics/train ELBO', elbo, epoch)
        writer.add_scalar('metrics/test log likelihood', ll, epoch)
        writer.add_scalar('metrics/test RMSE',  aux, epoch)
        writer.add_scalar('metrics/homoskedastic noise', torch.exp(0.5*loss.log_var).cpu().detach().numpy(), epoch)

        if epoch % 100 == 0:
            print(f'\rEpoch {epoch:4.0f}, '
              f'elbo: {elbo:10.4f}, '
              f'KL term: {kl:10.4f}, '
              f'train log likelihood term: {log_lik:8.4f}, '
              f'test log likelihood: {ll:8.4f}, '
              f'test auxilary: {aux:8.4f}, '
              f'train time: {train_time:6.4f}, '
              f'test time: {test_time:6.4f}')

        if epoch % 1000 == 0:

            predictions_train, actuals_train = inference.predictions(train=True)
            predictions_train = np.mean(predictions_train, 0)

            predictions_test, actuals_test = inference.predictions(train=False)
            predictions_test = np.mean(predictions_test, 0)

            plt.figure()
            plt.scatter(actuals_train, predictions_train)
            plt.scatter(actuals_test, predictions_test)
            plt.legend(['Train', 'Test'])
            plt.xlabel('actuals')
            plt.ylabel('predictions')
            plt.title(f'epcoh {epoch}')
            plt.plot([min(actuals_train), max(actuals_train)], [min(actuals_train), max(actuals_train)], color='r')
            plt.savefig(f'{fig_dir}/{epoch}.png')
            plt.close()

    writer.close()


if __name__ == '__main__':
    main()