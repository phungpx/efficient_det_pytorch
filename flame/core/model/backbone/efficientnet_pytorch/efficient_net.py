import torch
from torch import nn
from typing import List
from .model import EfficientNet


class _EfficientNet(nn.Module):
    def __init__(self, backbone_name: str = 'efficientnet-b0', pretrained: bool = False) -> None:
        super(_EfficientNet, self).__init__()
        if pretrained:
            model = EfficientNet.from_pretrained(backbone_name)
        else:
            model = EfficientNet.from_name(backbone_name)

        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc

        self.model = model

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        '''
        args:
            x: Tensor B x C x H x W
        outpus:
            feature_maps: List[Tensor]
            . P1: B, C1, H / 2 ^ 1, W / 2 ^ 1
            . P2: B, C2, H / 2 ^ 2, W / 2 ^ 2
            . P3: B, C3, H / 2 ^ 3, W / 2 ^ 3
            . P4: B, C4, H / 2 ^ 4, W / 2 ^ 4
            . P5: B, C5, H / 2 ^ 5, W / 2 ^ 5
        '''
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)

        feature_maps = []

        last_x = None
        for i, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(i) / len(self.model._blocks)

            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif i == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x

        del last_x

        return feature_maps


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name of backbone', type=str, default='efficientnet-b0')
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()

    names = [
        'efficientnet-b0',
        'efficientnet-b1',
        'efficientnet-b2',
        'efficientnet-b3',
        'efficientnet-b4',
        'efficientnet-b5',
        'efficientnet-b6',
        'efficientnet-b7',
        'efficientnet-b8',
        # Support the construction of 'efficientnet-l2' without pretrained weights
        'efficientnet-l2'
    ]
    if args.name not in names:
        print(f'{name} is invalid.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = _EfficientNet(backbone_name=args.name, pretrained=args.pretrained).to(device)

    dummy_input = torch.rand(size=[1, 3, 512, 512], dtype=torch.float32, device=device)

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

    print(f"Features Channels - {args.name}:")
    print({f'C{i}': feature.shape[1] for i, feature in enumerate(features, 1)})
