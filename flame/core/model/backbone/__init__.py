from .resnet import ResNet
from .densenet import DenseNet
from .efficientnet.efficient_net import _EfficientNet
from typing import Optional


def load_backbone(backbone_name: str = 'efficientnet-b0', pretrained: bool = False, num_layers: Optional[int] = None):
    resnet_layers_channels = {
        'resnet18': {'C1': 64, 'C2': 64, 'C3': 128, 'C4': 256, 'C5': 512},
        'resnet34': {'C1': 64, 'C2': 64, 'C3': 128, 'C4': 256, 'C5': 512},
        'resnet50': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1024, 'C5': 2048},
        'resnet101': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1024, 'C5': 2048},
        'resnet152': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1024, 'C5': 2048},
    }
    densenet_layers_channels = {
        'densenet121': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1024, 'C5': 1024},
        'densenet161': {'C1': 96, 'C2': 384, 'C3': 768, 'C4': 2112, 'C5': 2208},
        'densenet169': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1280, 'C5': 1664},
        'densenet201': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1792, 'C5': 1920},
    }
    efficientnet_layers_channels = {
        'efficientnet-b0': {'C1': 16, 'C2': 24, 'C3': 40, 'C4': 112, 'C5': 320},
        'efficientnet-b1': {'C1': 16, 'C2': 24, 'C3': 40, 'C4': 112, 'C5': 320},
        'efficientnet-b2': {'C1': 16, 'C2': 24, 'C3': 48, 'C4': 120, 'C5': 352},
        'efficientnet-b3': {'C1': 24, 'C2': 32, 'C3': 48, 'C4': 136, 'C5': 384},
        'efficientnet-b4': {'C1': 24, 'C2': 32, 'C3': 56, 'C4': 160, 'C5': 448},
        'efficientnet-b5': {'C1': 24, 'C2': 40, 'C3': 64, 'C4': 176, 'C5': 512},
        'efficientnet-b6': {'C1': 32, 'C2': 40, 'C3': 72, 'C4': 200, 'C5': 576},
        'efficientnet-b7': {'C1': 32, 'C2': 48, 'C3': 80, 'C4': 224, 'C5': 640},
        'efficientnet-b8': {'C1': 32, 'C2': 56, 'C3': 88, 'C4': 248, 'C5': 704},
        # Support the construction of 'efficientnet-l2' without pretrained weights
        'efficientnet-l2': {'C1': 72, 'C2': 104, 'C3': 176, 'C4': 480, 'C5': 1376},
    }

    if backbone_name in resnet_layers_channels:
        backbone = ResNet(backbone_name, pretrained=pretrained)
        layers_channels = resnet_layers_channels[backbone_name]
    elif backbone_name in densenet_layers_channels:
        backbone = DenseNet(backbone_name, pretrained=pretrained)
        layers_channels = densenet_layers_channels[backbone_name]
    elif backbone_name in efficientnet_layers_channels:
        backbone = _EfficientNet(backbone_name, pretrained=pretrained)
        layers_channels = efficientnet_layers_channels[backbone_name]
    else:
        raise ValueError(f'Not supported backbone {backbone_name}')

    if num_layers:
        layers_channels = layers_channels[:num_layers]

    return backbone, layers_channels
