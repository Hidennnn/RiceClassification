import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    def __init__(self, data_path, labels, batch_size=32, n_channels=1,
                 n_classes=5, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.data_paths = data_path
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
        x = []
        y = []

        for i, path in enumerate(temp_data_paths):
            array_to_append = np.load(path, allow_pickle=True)
            array_to_append = self.scaler.transform(array_to_append)
            x.append(array_to_append)

            # Store class
            y.append(self.labels[path])

        return x, to_categorical(np.array(y), num_classes=self.n_classes)
