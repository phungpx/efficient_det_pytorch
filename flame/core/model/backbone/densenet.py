import torch
from torch import nn
from torchvision import models
from typing import List


class DenseNet(nn.Module):
    def __init__(self, backbone_name: str = 'densenet121', pretrained: bool = False):
        super(DenseNet, self).__init__()
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        # layer1
        self.conv0 = backbone.features.conv0
        self.norm0 = backbone.features.norm0
        self.relu0 = backbone.features.relu0
        # layer2
        self.pool0 = backbone.features.pool0
        self.denseblock1 = backbone.features.denseblock1
        # layer3
        self.transition1 = backbone.features.transition1
        self.denseblock2 = backbone.features.denseblock2
        # layer4
        self.transition2 = backbone.features.transition2
        self.denseblock3 = backbone.features.denseblock3
        # layer5
        self.transition3 = backbone.features.transition3
        self.denseblock4 = backbone.features.denseblock4
        self.norm5 = backbone.features.norm5

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
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        features.append(x)
        # layer1
        x = self.pool0(x)
        x = self.denseblock1(x)
        features.append(x)
        # layer2
        x = self.transition1(x)
        x = self.denseblock2(x)
        features.append(x)
        # layer3
        x = self.transition2(x)
        x = self.denseblock3(x)
        features.append(x)
        # layer4
        x = self.transition3(x)
        x = self.denseblock4(x)
        x = self.norm5(x)
        features.append(x)

        return features


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', help='version of backbone', type=str, default='densenet121')
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()

    versions = ['densenet121', 'densenet161', 'densenet169', 'densenet201']
    if args.version not in versions:
        print(f'{args.version} is invalid.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = DenseNet(backbone_name=args.version, pretrained=args.pretrained).to(device)

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
