import torch
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import MFVI as mm

from tensorboardX import SummaryWriter

cuda = torch.cuda.is_available()


def evaluate_bayes_regression(inference, epochs, log_freq=100, log_dir='logs', verbose=False):

    writer = SummaryWriter(log_dir=log_dir)
    # writer.add_graph(inference)

    fig_dir = f'{log_dir}/figs'

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    elbos = []
    test_ll = []
    auxiliary = []
    noise_sigma = []
    train_ll = []
    train_kl = []

    for epoch in range(epochs):
        if (epoch == 0) & verbose:
            test_log_lik, _, test_auxiliary = inference.evaluate()
            print(f'Initial: {epoch:4.0f} test log likelihood: {test_log_lik:8.4f}, test rmse: {test_auxiliary:8.4f}')


        t = time.time()
        elbo, train_log_lik, kl = inference.train_step()
        train_time = time.time() - t
        test_log_lik, _, test_auxiliary = inference.evaluate()
        test_time = time.time() - train_time - t

        sy = torch.exp(0.5 * inference.loss.log_var).cpu().detach().numpy().tolist()[0]

        elbos.append(elbo)
        test_ll.append(test_log_lik)
        auxiliary.append(test_auxiliary)
        noise_sigma.append(sy)
        train_ll.append(train_log_lik)
        train_kl.append(kl)

        writer.add_scalar('metrics/train kl divergence', kl, epoch)
        writer.add_scalar('metrics/train log likelihood', train_log_lik, epoch)
        writer.add_scalar('metrics/train ELBO', elbo, epoch)
        writer.add_scalar('metrics/test log likelihood', test_log_lik, epoch)
        writer.add_scalar('metrics/test auxiliary (RMSE)', test_auxiliary, epoch)
        writer.add_scalar('parameters/homoskedastic noise', sy, epoch)

        if (epoch % log_freq == 0) & verbose:
            print(f'\rEpoch {epoch:4.0f}, \t ELBO: {elbo:10.4f}, \t KL term: {kl:10.4f}, \t train log likelihood term: {train_log_lik:8.4f}, \t test log likelihood: {test_log_lik:8.4f}, \t test auxiliary: {test_auxiliary:8.4f}, \t noise sigma: {sy:8.4f}, \t train time: {train_time:6.4f}, \t test time: {test_time:6.4f}')

        if epoch % (log_freq * 10) == 0:
            predictions_train, target_train = inference.predictions(train=True)
            predictions_train = np.mean(predictions_train, 0)

            predictions_test, target_test = inference.predictions(train=False)
            predictions_test = np.mean(predictions_test, 0)

            plt.figure()
            plt.scatter(target_train, predictions_train)
            plt.scatter(target_test, predictions_test)
            plt.legend(['Train', 'Test'])
            plt.xlabel('targets')
            plt.ylabel('predictions')
            plt.title(f'epcoh {epoch}')
            plt.plot([min(target_train), max(target_train)], [min(target_train), max(target_train)], color='r')
            plt.savefig(f'{fig_dir}/{epoch}.png')
            plt.close()

    writer.close()

    return {'elbo': elbos, 'test_ll': test_ll, 'test_auxiliary': auxiliary, 'noise_sigma': noise_sigma,
               'train_ll': train_ll,'train_kl': train_kl}


def evaluate_point_regression(model_trainer, epochs, log_freq=100, log_dir='logs', verbose=False):

    writer = SummaryWriter(log_dir=log_dir)
    # writer.add_graph(inference)

    fig_dir = f'{log_dir}/figs'
    os.mkdir(fig_dir)

    test_losses = []
    train_losses = []

    for epoch in range(epochs):
        if (epoch == 0) & verbose:
            test_loss = model_trainer.evaluate()
            print(f'Initial: {epoch:4.0f} test loss: {test_loss:8.4f}')


        t = time.time()
        test_loss = model_trainer.train_step()
        train_time = time.time() - t
        train_loss = model_trainer.evaluate()
        test_time = time.time() - train_time - t

        test_losses.append(test_loss)
        train_losses.append(train_loss)

        writer.add_scalar('metrics/train auxiliary (RMSE)', train_loss, epoch)
        writer.add_scalar('metrics/test auxiliary (RMSE)', test_loss, epoch)


        if (epoch % log_freq == 0) & verbose:
            print(f'\rEpoch {epoch:4.0f}, \t train loss: {train_loss:10.4f}, \t test loss: {test_loss:10.4f}')

        if epoch % (log_freq * 10) == 0:
            predictions_train, target_train = model_trainer.predictions(train=True)

            predictions_test, target_test = model_trainer.predictions(train=False)

            plt.figure()
            plt.scatter(target_train, predictions_train)
            plt.scatter(target_test, predictions_test)
            plt.legend(['Train', 'Test'])
            plt.xlabel('targets')
            plt.ylabel('predictions')
            plt.title(f'epcoh {epoch}')
            plt.plot([min(target_train), max(target_train)], [min(target_train), max(target_train)], color='r')
            plt.savefig(f'{fig_dir}/{epoch}.png')
            plt.close()

    writer.close()

    return {'train_loss': train_losses, 'test_loss': test_losses}
