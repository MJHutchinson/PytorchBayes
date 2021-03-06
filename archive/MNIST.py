import time
import os
import torch
import MFVI as mm
import torch.optim as optim
from model.MLP_MFVI import MLP_MFVI
from data.data_sets import ClassificationDataloader


def main():
    use_cuda = torch.cuda.is_available()
    batch_size = 900
    test_batch_size = 900
    epochs = 2

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=test_batch_size, shuffle=True, **kwargs)

    # train_loader = torch.utils.data.DataLoader(
    #     ClassificationDataset('mnist', train=True),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     **kwargs
    # )
    #
    # test_loader = torch.utils.data.DataLoader(
    #     ClassificationDataset('mnist', train=False),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     **kwargs
    # )

    train_loader = ClassificationDataloader('mnist', batch_size, data_dir=os.path.abspath('../data'), train=True, shuffle=True)
    test_loader = ClassificationDataloader('mnist', batch_size, data_dir=os.path.abspath('../data'), train=False, shuffle=False)

    model = MLP_MFVI(28 * 28, [400, 400], 10)

    if use_cuda:
        model.cuda()

    loss = mm.CrossEntropyLoss()

    optimiser = optim.Adam([{'params':model.parameters()}, {'params': loss.parameters()}])

    inference = mm.MeanFieldVariationalInference(model, loss, optimiser, train_loader, test_loader)

    for epoch in range(epochs):
        t = time.time()
        elbo, log_lik, kl = inference.train_step()
        train_time = time.time() - t
        ll, kl, aux = inference.evaluate()
        test_time = time.time() - train_time - t

        print(f'\rEpoch {epoch:4.0f}, '
              f'elbo: {elbo:10.4f}, '
              f'KL term: {kl:10.4f}, '
              f'train log likelihood term: {log_lik:8.4f}, '
              f'test log likelihood: {ll:8.4f}, '
              f'test auxilary: {aux:8.4f}, '
              f'train time: {train_time:6.4f}, '
              f'test time: {test_time:6.4f}')


if __name__ == '__main__':
    main()