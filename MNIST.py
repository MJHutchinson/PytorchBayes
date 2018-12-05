from MFVI.model.MLP import MFVIClassificationMLP
from MFVI.inference.MFVI import MeanFieldVariationalInference
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import time


def main():
    use_cuda = torch.cuda.is_available()
    batch_size = 100
    device = torch.device("cuda" if use_cuda else "cpu")
    epochs = 300

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = MFVIClassificationMLP(28*28, [400, 400], 10)
    model = model.to(device)

    inference = MeanFieldVariationalInference(model, optim.Adam(model.parameters()))

    for epoch in range(epochs):
        t = time.time()
        elbo, log_lik, kl = inference.step(train_loader, 60000, 10, device)
        train_time = time.time() - t
        ll, kl, aux = inference.evaluate(test_loader, 10000, 100, device)
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