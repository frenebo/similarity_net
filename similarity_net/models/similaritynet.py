from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Concatenate, Activation

from .backbones.backbone import Backbone

def mesh_backbone_outputs_and_return_prob(first_bb_outputs, second_bb_outputs):
    assert len(first_bb_outputs) == len(second_bb_outputs), "Should be same number of first and of second backbone outputs"

    bb_combined_layers = []

    for i, (first_output, second_output) in enumerate(zip(first_bb_outputs, second_bb_outputs)):
        # print("first shape", first_output.shape)
        # print("second shape", second_output.shape)
        assert first_output.shape[-1] == second_output.shape[-1], "First and second outputs should have same number of filters, even if not same width/height"
        assert len(first_output.shape) == 4, "First output should be 2D"
        assert len(second_output.shape) == 4, "Second output should be 2D"

        shared_conv_layer = Conv2D(
            filters=10,
            kernel_size=9,
            padding="same",
            activation="relu",
            name="first_shared_conv_{}".format(i)
        )

        shared_pooling_layer = GlobalAveragePooling2D(name="shared_pooling_{}".format(i))

        first = first_output
        first = shared_conv_layer(first)
        first = shared_pooling_layer(first)

        second = second_output
        second = shared_conv_layer(second)
        second = shared_pooling_layer(second)

        # dense_combined = Dense()
        x = Concatenate(axis=-1, name="concat_{}".format(i))([first, second])
        x = Dense(units=10, name="dense_{}".format(i), activation="tanh")(x)

        bb_combined_layers.append(x)

    x = Concatenate(axis=-1, name="concat_layers")(bb_combined_layers)
    # x = Dense(units=10, name="first_dense_after_concat", activation="tanh")(x)
    x = Dense(units=1, name="second_dense_after_concat", activation="tanh")(x)
    # x = Activation("softmax")(x)

    return x

def create_similaritynet(backbone):
    assert isinstance(backbone, Backbone)

    first_input = Input((None, None, 3))
    second_input = Input((None, None, 3))

    first_bb_outputs = backbone.call_on_inputs(first_input)
    second_bb_outputs = backbone.call_on_inputs(second_input)

    prob = mesh_backbone_outputs_and_return_prob(first_bb_outputs, second_bb_outputs)

    model = Model(inputs=[first_input, second_input], outputs=prob)

    return model
