import os
import pickle
import torch
import torch.utils.data as data
import numpy as np


class ClassificationDataset(data.Dataset):

    def __init__(self, pickle_name, data_dir='./data', train=True):
        self.pickle_name = pickle_name
        self.train = train

        f = open(f'{data_dir}/{self.pickle_name}.pkl', 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        if self.train:
            self.X_train = np.vstack((train_set[0], valid_set[0]))
            self.Y_train = np.hstack((train_set[1], valid_set[1])).astype(np.int)
            self.classes = int(np.max(self.Y_train) + 1)
        else:
            self.X_test = test_set[0]
            self.Y_test = test_set[1].astype(np.int)
            self.classes = int(np.max(self.Y_test) + 1)

    def __getitem__(self, index):
        if self.train:
            return self.X_train[index], self.Y_train[index]
        else:
            return self.X_test[index], self.Y_test[index]

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test)


class ClassificationDataloader():
    def __init__(self, pickle_name, batch_size, data_dir='./data_dir', train=True, shuffle=False):
        self.pickle_name = pickle_name
        self.batch_size = batch_size
        self.train = train
        self.shuffle = shuffle

        f = open(f'{data_dir}/{self.pickle_name}.pkl', 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        if self.train:
            self.X = np.vstack((train_set[0], valid_set[0]))
            self.Y = np.hstack((train_set[1], valid_set[1])).astype(np.int)
            self.length = np.ceil(len(self.X)/self.batch_size)
            self.data_size = len(self.X)
        else:
            self.X = test_set[0]
            self.Y = test_set[1].astype(np.int)
            self.length = np.ceil(len(self.X) / self.batch_size)
            self.data_size = len(self.X)

    def __len__(self):
        if self.train:
            return self.length

    def __iter__(self):
        return ClassificationDataloaderIter(self)


class ClassificationDataloaderIter(object):
    def __init__(self, dataloader):
        self.X = dataloader.X
        self.Y = dataloader.Y
        self.batch_size = dataloader.batch_size
        self.length = dataloader.length
        self.data_size = dataloader.data_size
        self.shuffle = dataloader.shuffle
        self.index = 0

        if self.shuffle:
            indices = list(range(self.X.shape[0]))
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.Y = self.Y[indices]

    def __next__(self):
        if self.index == self.length:
            raise StopIteration

        start_ind = self.index * self.batch_size
        end_ind = np.min([(self.index + 1) * self.batch_size, self.data_size])

        self.index += 1

        return (torch.Tensor(self.X[start_ind:end_ind, :]), torch.LongTensor(self.Y[start_ind:end_ind]))

    def __len__(self):
        return self.length



class RegressionDataloader():
    def __init__(self, pickle_name, batch_size, data_dir='./data_dir', train=True, shuffle=False):
        self.pickle_name = pickle_name
        self.batch_size = batch_size
        self.train = train
        self.shuffle = shuffle

        f = open(f'{data_dir}/{self.pickle_name}.pkl', 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        x_train = np.vstack((train_set[0], valid_set[0]))
        y_train = np.hstack((train_set[1], valid_set[1])).astype(np.int)

        self.X_means = 0.5 * (
                    np.expand_dims(np.max(x_train, axis=0), 0) + np.expand_dims(np.min(x_train, axis=0),
                                                                                     0))  # np.expand_dims(np.mean(self.X_train, axis=0), 0)
        self.Y_means = 0.5 * (
                    np.expand_dims(np.max(y_train, axis=0), 0) + np.expand_dims(np.min(y_train, axis=0),
                                                                                     0))  # np.expand_dims(np.mean(self.Y_train, axis=0), 0)
        self.X_sigmas = 0.5 * (
                    np.expand_dims(np.max(x_train, axis=0), 0) - np.expand_dims(np.min(x_train, axis=0),
                                                                                     0))  # np.sqrt(np.expand_dims(np.var(self.X_train, axis=0), 0)) #
        self.Y_sigmas = 0.5 * (
                    np.expand_dims(np.max(y_train, axis=0), 0) - np.expand_dims(np.min(y_train, axis=0),
                                                                                     0))  # np.sqrt(np.expand_dims(np.var(self.Y_train, axis=0), 0)) #

        if self.train:
            self.X_orig = np.vstack((train_set[0], valid_set[0]))
            self.Y_orig = np.hstack((train_set[1], valid_set[1])).astype(np.int)
            self.X, self.Y = self.transform(self.X_orig, self.Y_orig)

            self.length = np.ceil(len(self.X)/self.batch_size)
            self.data_size = len(self.X)
        else:
            self.X_orig = test_set[0]
            self.Y_orig = test_set[1].astype(np.int)
            self.X, self.Y = self.transform(self.X_orig, self.Y_orig)

            self.length = np.ceil(len(self.X) / self.batch_size)
            self.data_size = len(self.X)

        if len(self.Y.shape) == 1:
            self.Y = np.expand_dims(self.Y, 1)

        self.classes = self.Y.shape[1]


    def transform(self, X, Y):
        return (X-self.X_means)/self.X_sigmas, (Y - self.Y_means)/self.Y_sigmas

    def antitransform(self, X, Y):
        return (X*self.X_sigmas) + self.X_means, (Y*self.Y_sigmas) + self.Y_means

    def get_dims(self):
        # Get data input and output dimensions
        return self.X.shape[1], self.X.shape[0], self.classes

    def get_transforms(self):
        return self.X_means, self.X_sigmas, self.Y_means, self.Y_sigmas

    def __len__(self):
        if self.train:
            return self.length

    def __iter__(self):
        return RegressionDataloaderIter(self)


class RegressionDataloaderIter(object):
    def __init__(self, dataloader):
        self.X = dataloader.X
        self.Y = dataloader.Y
        self.batch_size = dataloader.batch_size
        self.length = dataloader.length
        self.data_size = dataloader.data_size
        self.shuffle = dataloader.shuffle
        self.index = 0

        if self.shuffle:
            indices = list(range(self.X.shape[0]))
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.Y = self.Y[indices]

    def __next__(self):
        if self.index == self.length:
            raise StopIteration

        start_ind = self.index * self.batch_size
        end_ind = np.min([(self.index + 1) * self.batch_size, self.data_size])

        self.index += 1

        return (torch.Tensor(self.X[start_ind:end_ind, :]), torch.Tensor(self.Y[start_ind:end_ind]))

    def __len__(self):
        return self.length


