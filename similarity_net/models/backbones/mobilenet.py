from keras.applications.mobilenet import MobileNet
from keras.models import Model
import keras

from .backbone import Backbone

class MobilenetBackbone(Backbone):
    def __init__(
        self,
        alpha=1.0,
    ):
        inputs = keras.layers.Input((None, None, 3))
        mnet = MobileNet(
            input_tensor=inputs,
            alpha=alpha,
            include_top=False,
            pooling=None,
            weights=None)
        backbone_layer_names = ['conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu']
        backbone_outputs = [mnet.get_layer(name).output for name in backbone_layer_names]

        self.backbone_model = Model(inputs=inputs, outputs=backbone_outputs, name=mnet.name)

        super(MobilenetBackbone, self).__init__()

    def call_on_inputs(self, inputs):
        return self.backbone_model(inputs)