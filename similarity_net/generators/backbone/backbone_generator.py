import keras
import numpy as np
import random

class BackboneGenerator(keras.utils.Sequence):
    def __init__(self, steps):
        self.steps = steps

        self.indices = list(self.num_items())

        self.on_epoch_end()

    def on_epoch_end(self):
        random.shuffle(self.indices)

    def __getitem__(self, idx):
        img, label = self.load_item(self.indices[idx])
        assert isinstance(img, np.ndarray)
        assert isinstance(label, int)
        return np.array([img]), np.array([label])

    def __len__(self):
        return self.steps

    def num_items(self):
        """ Number of training items available
        """
        raise NotImplementedError()

    def load_item(self, idx):
        """ Returns image np array along with label integer
        """
        raise NotImplementedError()
