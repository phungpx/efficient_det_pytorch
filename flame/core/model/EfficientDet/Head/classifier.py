import torch
from torch import nn
from typing import Tuple, List
from .utils import SeparableConvBlock, MemoryEfficientSwish, Swish


class Classifier(nn.Module):
    def __init__(self, n_anchors: int = 9,
                 n_classes: int = 80,
                 compound_coef: int = 0,
                 D_class: List[int] = [3, 3, 3, 4, 4, 4, 5, 5, 5],
                 W_pred: List[int] = [64, 88, 112, 160, 224, 288, 384, 384, 384],
                 onnx_export: bool = False):
        super(Classifier, self).__init__()
        self.n_anchors = n_anchors
        self.n_classes = n_classes

        n_layers = D_class[compound_coef]
        n_channels = W_pred[compound_coef]
        n_pyramid_levels = 5 if compound_coef <= 7 else 6  # D0 - D7: 5 pyramid levels, D7x: 6 pyramid levels

        self.separable_convs = nn.ModuleList(
            [SeparableConvBlock(in_channels=n_channels, out_channels=n_channels,
                                use_batchnorm=False, use_activation=False,
                                onnx_export=onnx_export) for _ in range(n_layers)]
        )

        self.batch_norms_levels = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(num_features=n_channels, momentum=0.01, eps=1e-3)
                            for _ in range(n_layers)])
             for _ in range(n_pyramid_levels)]
        )

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.head_conv = SeparableConvBlock(in_channels=n_channels,
                                            out_channels=n_anchors * n_classes,
                                            use_batchnorm=False,
                                            use_activation=False,
                                            onnx_export=onnx_export)

    def forward(self, pyramid_features: Tuple[torch.Tensor]) -> torch.Tensor:
        '''
        args:
            pyramid_features: (P3', P4', P5', P6', P7')
                P3': B, W_pred, input / 2 ^ 3, input / 2 ^ 3
                P4': B, W_pred, input / 2 ^ 4, input / 2 ^ 4
                P5': B, W_pred, input / 2 ^ 5, input / 2 ^ 5
                P6': B, W_pred, input / 2 ^ 6, input / 2 ^ 6
                P7': B, W_pred, input / 2 ^ 7, input / 2 ^ 7
        outputs:
            x: Tensor [B, (H3 * W3 * n_anchors
                           + H4 * W4 * n_anchors
                           + H5 * W5 * n_anchors
                           + H6 * W6 * n_anchors
                           + H7 * W7 * n_anchors), n_classes]
        '''
        features = []
        for x, batch_norms_level in zip(pyramid_features, self.batch_norms_levels):
            for separable_conv, batch_norm in zip(self.separable_convs, batch_norms_level):
                x = separable_conv(x)
                x = batch_norm(x)
                x = self.swish(x)

            x = self.head_conv(x)

            x = x.permute(0, 2, 3, 1).contiguous()

            B, H, W, C = x.shape
            x = x.view(B, H, W, self.n_anchors, self.n_classes).contiguous()
            x = x.view(B, H * W * self.n_anchors, self.n_classes).contiguous()

            features.append(x)

        features = torch.cat(features, dim=1)
        x = features.sigmoid()

        return x
