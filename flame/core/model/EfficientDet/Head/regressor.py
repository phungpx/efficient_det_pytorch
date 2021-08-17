import torch
from torch import nn
from typing import Tuple, List
from .utils import SeparableConvBlock


class Regressor(nn.Module):
    def __init__(self, n_anchors: int, compound_coef: int,
                 D_box: List[int], W_bifpn: List[int], onnx_export: bool = False) -> None:
        '''
        Args:
            n_anchors = num_scales * num_aspect_ratios
            compound_coef: efficient det version
            D_box:
            W_bifpn:
        '''
        super(Regressor, self).__init__()
        self.n_layers = D_box[compound_coef]

        n_channels = W_bifpn[compound_coef]

        self.separable_conv = SeparableConvBlock(in_channels=n_channels,
                                                 out_channels=n_channels,
                                                 use_batchnorm=True,
                                                 use_activation=True,
                                                 onnx_export=onnx_export)

        self.head_conv = SeparableConvBlock(in_channels=n_channels,
                                            out_channels=n_anchors * 4,
                                            use_batchnorm=False,
                                            use_activation=False,
                                            onnx_export=onnx_export)

    def forward(self, inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        '''
        args:
            inputs: (P3', P4', P5', P6', P7')
                P3': B, W_bifpn, input / 2 ^ 3, input / 2 ^ 3
                P4': B, W_bifpn, input / 2 ^ 4, input / 2 ^ 4
                P5': B, W_bifpn, input / 2 ^ 5, input / 2 ^ 5
                P6': B, W_bifpn, input / 2 ^ 6, input / 2 ^ 6
                P7': B, W_bifpn, input / 2 ^ 7, input / 2 ^ 7
        outputs:
            x: Tensor [B, (H3 * W3 * n_anchors
                           + H4 * W4 * n_anchors
                           + H5 * W5 * n_anchors
                           + H6 * W6 * n_anchors
                           + H7 * W7 * n_anchors), 4]
        '''
        features = []
        for x in inputs:
            for _ in self.n_layers:
                x = self.separable_conv(x)

            x = self.head_conv(x)

            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.shape[0], -1, 4).contiguous()

            features.append(x)

        x = torch.cat(features, dim=1)

        return x
