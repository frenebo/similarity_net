from .mobilenet import MobilenetBackbone
from .densenet import DenseNetBackbone

def get_backbone_class(backbone_name):
    if "mobilenet" in backbone_name:
        return MobilenetBackbone
    elif "densenet" in backbone_name:
        return DenseNetBackbone
    else:
        raise ValueError("No backbone with name '{}'".format(backbone_name))