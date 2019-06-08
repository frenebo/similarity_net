from keras.models import Model
from keras.layer import Input

def similaritynet(backbone_model):
    first_input = Input((None, None, 3))
    second_input = Input((None, None, 3))

    first_input_ = backbone_model(first_input)
    second_through_backbone = backbone_model(second_input)

