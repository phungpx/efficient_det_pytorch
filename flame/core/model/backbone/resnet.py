import torch
from torch import nn
from torchvision import models
from typing import List


class ResNet(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = False):
        super(ResNet, self).__init__()
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        # layer0
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        # layer1
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        # layer2
        self.layer2 = backbone.layer2
        # layer3
        self.layer3 = backbone.layer3
        # layer4
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        '''
            Args:
                x: B x 3 x H x W
            Output:
                List[torch.Tensor]
                # C1: B x C1 x (H / 2 ^1) x (W / 2 ^ 1)
                # C2: B x C2 x (H / 2 ^2) x (W / 2 ^ 2)
                # C3: B x C3 x (H / 2 ^3) x (W / 2 ^ 3)
                # C4: B x C4 x (H / 2 ^4) x (W / 2 ^ 4)
                # C5: B x C5 x (H / 2 ^5) x (W / 2 ^ 5)
        '''
        features = []
        # layer0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        # layer1
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        # layer2
        x = self.layer2(x)
        features.append(x)
        # layer3
        x = self.layer3(x)
        features.append(x)
        # layer4
        x = self.layer4(x)
        features.append(x)

        return features


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', help='version of backbone', type=str, default='resnet18')
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()

    versions = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    if args.version not in versions:
        print(f'{version} is invalid.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = ResNet(backbone_name=args.version, pretrained=args.pretrained).to(device)

    dummy_input = torch.rand(size=[1, 3, 224, 224], dtype=torch.float32, device=device)

    with torch.no_grad():
        t1 = time.time()
        features = backbone(dummy_input)
        t2 = time.time()

    print(f"Input Shape: {dummy_input.shape}")
    print(f"Number of parameters: {sum((p.numel() for p in backbone.parameters() if p.requires_grad))}")
    print(f"Processing Time: {t2 - t1}s")
    print(f"Features Shape:")
    for i, feature in enumerate(features, 0):
        print(f'Layer: {i} - Shape: {feature.shape}')

    print(f'Features Channels - {args.version}: {[feature.shape[1] for feature in features]}')
