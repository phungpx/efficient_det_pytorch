import torch
from torch import nn
from typing import List
from .model import EfficientNet


class _EfficientNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = 'efficientnet-b0',
        pretrained: bool = False
    ) -> None:
        super(_EfficientNet, self).__init__()

        model = EfficientNet.from_pretrained(backbone_name, pretrained)
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
