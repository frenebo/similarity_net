import keras
from keras.applications import densenet
from .backbone import Backbone

allowed_versions = {
    '121': ([6, 12, 24, 16], densenet.DenseNet121),
    '169': ([6, 12, 32, 32], densenet.DenseNet169),
    '201': ([6, 12, 48, 32], densenet.DenseNet201),
}

class DenseNetBackbone(Backbone):
    def __init__(self, version="121"):
        inputs = keras.layers.Input((None, None, 3))
        try:
            blocks = allowed_versions[version][0]
            net_class = allowed_versions[version][1]
            net = net_class(input_tensor=inputs)
        except KeyError:
            possible_versions = list(possible_versions.keys())
            raise ValueError("Invalid Densenet version '{}'. Allowed versions: {}".format(net_version, possible_versions))

        bb_layer_names = ["conv{}_block{}_concat".format(idx + 2, block_num) for idx, block_num in enumerate(blocks)]
        bb_layer_outputs = [net.get_layer(layer_name).output for layer_name in bb_layer_names]

        self.backbone_model = keras.models.Model(inputs=inputs, outputs=bb_layer_outputs)

        super(DenseNetBackbone, self).__init__()

    @staticmethod
    def from_weights(weights_path):
        bb = DenseNetBackbone()
        bb.backbone_model.load_weights(weights_path)

        return bb

    def call_on_inputs(self, inputs):
        return self.backbone_model(inputs)