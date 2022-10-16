import numpy as np
from math import ceil

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__
        self.indexes = np.array(range(self.X.shape[0]))

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return ceil(self.X.shape[0] / self.batch_size)  # drop_last = False

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        self.batch_id = 0
        if self.shuffle:
            self.indexes = np.array(range(self.X.shape[0]))
            np.random.shuffle(self.indexes)
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        if self.batch_id < len(self):
            batch_indexes = self.indexes[self.batch_id * self.batch_size : (self.batch_id + 1) * self.batch_size]
            X_batch = self.X[batch_indexes]
            y_batch = self.y[batch_indexes]
            self.batch_id += 1
            return X_batch, y_batch
        raise StopIteration
