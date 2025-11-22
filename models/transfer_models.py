import torch.nn as nn
from torchvision import models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_resnet18(num_classes=5, feature_extract=True, weights=models.ResNet18_Weights.DEFAULT):
    model = models.resnet18(weights=weights)

    # заморозити всі параметри якщо feature_extract=True
    set_parameter_requires_grad(model, feature_extract)

    # замінюємо останній шар на свій
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def get_mobilenet_v2(num_classes=5, feature_extract=True, weights=models.ResNet18_Weights.DEFAULT):
    model = models.mobilenet_v2(weights=weights)

    set_parameter_requires_grad(model, feature_extract)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
