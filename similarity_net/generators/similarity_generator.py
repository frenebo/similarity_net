import keras

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

        self.initialize(self.batch_size, self.steps_per_epoch, self.proportion_matching)

    def on_epoch_end(self):
        self.initialize(self.batch_size, self.steps_per_epoch, self.proportion_matching)

    def initialize_batches(self, batch_size, pair_count):
        raise NotImplementedError()

    def get_input_output(self, index):
        raise NotImplementedError()

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        inputs, targets = self.get_input_output(index)
    #     """
    #     Number of batches for generator.
    #     """

    #     return len(self.groups)

    # def __getitem__(self, index):
    #     """
    #     Keras sequence method for generating batches.
    #     """
    #     group = self.groups[index]
    #     inputs, targets = self.compute_input_output(group)

    #     return inputs, targets
