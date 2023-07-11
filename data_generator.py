import numpy as np
import cv2
from tensorflow.keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    def __init__(self, data_path, labels, batch_size=32, dim=(250,250), n_channels=3,
                 n_classes=5, shuffle=True):
        self.data_paths = data_path
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        temp_data_paths = [self.data_paths[k] for k in indexes]

        x, y = self.__data_generation(temp_data_paths)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_data_paths):
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        for i, path in enumerate(temp_data_paths):
            x[i,] = cv2.imread(path)
            y[i] = self.labels[path]

        return x, to_categorical(y, num_classes=self.n_classes)
