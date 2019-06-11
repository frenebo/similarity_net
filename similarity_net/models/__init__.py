import keras.models
from . import backbones
from .similaritynet import create_similaritynet

def load_model(model_path, backbone_name):
    BackboneClass = backbones.get_backbone_class(backbone_name)

    custom_objects = {}
    custom_objects.update(BackboneClass.custom_objects)

    return keras.models.load_model(model_path, custom_objects=custom_objects)

# def save_model(model, model_):


def instantiate_model(backbone_name):
    BackboneClass = backbones.get_backbone_class(backbone_instance)
    backbone = BackboneClass()

    return create_similaritynet(backbone)