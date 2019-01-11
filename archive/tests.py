from data.data_sets import ClassificationDataset, ClassificationDataloader, RegressionDataloader
import torch
import numpy as np
import time
from torchvision import datasets, transforms

use_cuda = torch.cuda.is_available()
loops = 1
batch_size = 900
test_batch_size = 900
epochs = 2
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

t=time.time()
for e in range(0, loops):
    for idx, (inputs, targets) in enumerate(train_loader):
        if idx == 0: t = time.time()
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs+inputs
print(time.time()-t)

t=time.time()
for e in range(0, loops):
    for idx, (inputs, targets) in enumerate(train_loader):
        if idx == 0: t = time.time()
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs+inputs
print(time.time()-t)

pytorch_loader = torch.utils.data.DataLoader(
        ClassificationDataset('mnist', train=True),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

t=time.time()
for e in range(0, loops):
    for idx, (inputs, targets) in enumerate(pytorch_loader):
        if idx == 0: t = time.time()
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs+inputs
print(time.time()-t)

numpy_loader = RegressionDataloader('yacht', data_dir='./data', batch_size=batch_size, train=True, shuffle=True)

x_train, y_train = numpy_loader.X, numpy_loader.Y

N = x_train.shape[0]
total_batch = int(np.ceil(N * 1.0 / batch_size))

t=time.time()
for e in range(0, loops):
    for i in range(total_batch):
        if i == 0: t = time.time()
        start_ind = i * batch_size
        end_ind = np.min([(i + 1) * batch_size, N])
        batch_x = x_train[start_ind:end_ind, :]
        batch_y = y_train[start_ind:end_ind]

        batch_x, batch_y = torch.Tensor(batch_x).cuda(), torch.Tensor(batch_y).cuda()
        batch_x = batch_x + batch_x
print(time.time()-t)


for e in range(0, loops):
    for idx, (inputs, targets) in enumerate(numpy_loader):
        if idx == 0: t=time.time()
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs+inputs
print(time.time()-t)