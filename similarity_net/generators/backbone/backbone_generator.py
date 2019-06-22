import keras
import numpy as np
import random

class BackboneGenerator(keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        steps_per_epoch,
        ):
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size

        self.indices = list(self.num_items())

        self.on_epoch_end()

    def on_epoch_end(self):
        random.shuffle(self.indices)

    def __getitem__(self, batch_idx):
        imgs = []
        labels = []

        start_idx = self.batch_size * batch_idx
        end_idx   = self.batch_size * (batch_idx + 1)

        for idx in range(start_idx, end_idx):
            print("idx: ", idx)

            img, label = self.load_item(self.indices[idx])

            assert isinstance(img, np.ndarray)
            assert isinstance(label, int)
            return np.array([img]), np.array([label])

            imgs.append(img)
            labels.append(label)

        return np.array(imgs), np.array(labels)

    def __len__(self):
        return self.steps_per_epoch

    def num_items(self):
        """ Number of training items available
        """
        raise NotImplementedError()

    def load_item(self, idx):
        """ Returns image np array along with label integer
        """
        raise NotImplementedError()
