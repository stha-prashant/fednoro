import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .model_resnet import ResNet18, ResNet34

def get_model(model_name, args):
    """Returns a CNN model
    Args:
      model_name: model name
      pretrained: True or False
    Returns:
      model: the desired model
    Raises:
      ValueError: If model name is not recognized.
    """
    if model_name == 'resnet18':
        return ResNet18(num_classes=args.num_classes)
    elif model_name == 'resnet34':
        return ResNet34(num_classes=args.num_classes)

def modify_last_layer(model_name, model, num_classes, normed=False, bias=True):
    """modify the last layer of the CNN model to fit the num_classes
    Args:
      model_name: model name
      model: CNN model
      num_classes: class number
    Returns:
      model: the desired model
    """

    if 'Resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = classifier(num_ftrs, num_classes)
        last_layer = model.fc
    else:
        raise NotImplementedError
    return model, last_layer


def classifier(num_features, num_classes):
    last_linear = nn.Linear(num_features, num_classes)
    return last_linear



def get_feature_length(model_name, model):
    """get the feature length of the last feature layer
    Args:
      model_name: model name
      model: CNN model
    Returns:
      num_ftrs: the feature length of the last feature layer
    """
    if 'Vgg' in model_name:
        num_ftrs = model.classifier._modules['6'].in_features
    elif 'Dense' in model_name:
        num_ftrs = model.classifier.in_features
    elif 'Resnet' in model_name:
        num_ftrs = model.fc.in_features
    elif 'Efficient' in model_name:
        num_ftrs = model._fc.in_features
    elif 'RegNet' in model_name:
        num_ftrs = model.head.fc.in_features
    else:
        num_ftrs = model.last_linear.in_features

    return num_ftrs