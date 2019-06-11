from .keras_mobilenet_backbone import KerasMobilenetBackbone

def get_backbone_class(backbone_name):
    if backbone_name == "keras_mobilenet":
        return KerasMobilenetBackbone
    else:
        raise ValueError("No backbone with name '{}'".format(backbone_name))