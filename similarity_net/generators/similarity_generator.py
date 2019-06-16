import keras
import numpy as np

class SimilarityGenerator(keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        steps_per_epoch,
        proportion_matching,
        ):
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.proportion_matching = proportion_matching

        # self.initialize(self.batch_size, self.steps_per_epoch, self.proportion_matching)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.initialize(self.batch_size, self.steps_per_epoch, self.proportion_matching)

    def initialize(self, batch_size, pair_count):
        raise NotImplementedError()

    def get_input_output(self, index):
        raise NotImplementedError()

    # @TODO is this right?
    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        inputs, targets = self.get_input_output(index)

        # Should be [first_image_batch, second_image_batch]
        assert isinstance(inputs, list)
        assert len(inputs) == 2
        assert isinstance(inputs[0], np.ndarray)
        assert isinstance(inputs[1], np.ndarray)

        # should be target score batch
        assert isinstance(targets, np.ndarray)

        # print(targets)



        return inputs, targets
