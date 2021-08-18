import torch
from torch import nn
from typing import List
from .model import EfficientNet


class EfficientNetBackBone(nn.Module):
    def __init__(self, compound_coef: int = 0, weight_path: str = None):
        super(EfficientNetBackBone, self).__init__()
        efficient_net = EfficientNet.from_pretrained(model_name=f'efficientnet-b{compound_coef}',
                                                     weights_path=weight_path)
        del efficient_net._conv_head
        del efficient_net._bn1
        del efficient_net._avg_pooling
        del efficient_net._dropout
        del efficient_net._fc

        self.model = efficient_net

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

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        # try recording stride changing when creating efficientnet, and then apply it here.
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
