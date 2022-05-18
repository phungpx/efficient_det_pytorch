import torch
from torch import nn
from typing import List
from .utils.basic_blocks import SeparableConvBlock, Swish, MemoryEfficientSwish


__all__ = ['Regressor', 'Classifier']


class Regressor(nn.Module):
    def __init__(
        self,
        BiFPN_out_channels: int,
        num_anchors: int,
        num_layers: int,
        num_pyramid_levels: int = 5,
        onnx_export: bool = False
    ):
        super(Regressor, self).__init__()
        self.conv_list = nn.ModuleList(
            [
                SeparableConvBlock(
                    in_channels=BiFPN_out_channels,
                    out_channels=BiFPN_out_channels,
                    use_batch_norm=False,
                    use_activation=False,
                    onnx_export=onnx_export
                )
                for _ in range(num_layers)
            ]
        )

        self.bn_list = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.BatchNorm2d(num_features=BiFPN_out_channels, momentum=0.01, eps=1e-3) for _ in range(num_layers)]
                )
                for _ in range(num_pyramid_levels)
            ]
        )

        self.header = SeparableConvBlock(
            in_channels=BiFPN_out_channels,
            out_channels=num_anchors * 4,
            use_batch_norm=False,
            use_activation=False,
            onnx_export=onnx_export,
        )
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, pyramid_features: List[torch.Tensor]) -> torch.Tensor:
        '''
            Args:
                pyramid_features: (P3, P4, P5, P6, P7)
                    P3: B, C_FPN, H / 2 ^ 3, W / 2 ^ 3
                    P4: B, C_FPN, H / 2 ^ 4, W / 2 ^ 4
                    P5: B, C_FPN, H / 2 ^ 5, W / 2 ^ 5
                    P6: B, C_FPN, H / 2 ^ 6, W / 2 ^ 6
                    P7: B, C_FPN, H / 2 ^ 7, W / 2 ^ 7
            Outputs:
                x: Tensor [B, (H3 * W3 * n_anchors
                               + H4 * W4 * n_anchors
                               + H5 * W5 * n_anchors
                               + H6 * W6 * n_anchors
                               + H7 * W7 * n_anchors), 4]
        '''
        features = []
        for x, batch_norm_layers in zip(pyramid_features, self.bn_list):
            for bn, conv in zip(batch_norm_layers, self.conv_list):
                x = self.swish(bn(conv(x)))

            x = self.header(x)
            x = x.permute(0, 2, 3, 1).contiguous()  # B x H x W x (num_anchors * 4)
            x = x.view(x.shape[0], -1, 4).contiguous()  # B x (H * W * num_anchors) x 4

            features.append(x)

        x = torch.cat(features, dim=1)

        return x


class Classifier(nn.Module):
    def __init__(
        self,
        BiFPN_out_channels: int,
        num_anchors: int,
        num_classes: int,
        num_layers: int,
        num_pyramid_levels: int = 5,
        onnx_export: bool = False
    ):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv_list = nn.ModuleList(
            [
                SeparableConvBlock(
                    in_channels=BiFPN_out_channels,
                    out_channels=BiFPN_out_channels,
                    use_batch_norm=False,
                    use_activation=False,
                    onnx_export=onnx_export
                )
                for _ in range(num_layers)
            ]
        )

        self.bn_list = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.BatchNorm2d(num_features=BiFPN_out_channels, momentum=0.01, eps=1e-3) for _ in range(num_layers)]
                )
                for j in range(num_pyramid_levels)
            ]
        )

        self.header = SeparableConvBlock(
            in_channels=BiFPN_out_channels,
            out_channels=num_anchors * num_classes,
            use_batch_norm=False,
            use_activation=False,
            onnx_export=onnx_export,
        )
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, pyramid_features: List[torch.Tensor]) -> torch.Tensor:
        '''
            Args:
                pyramid_features: (P3, P4, P5, P6, P7)
                    P3: B, C_FPN, H / 2 ^ 3, W / 2 ^ 3
                    P4: B, C_FPN, H / 2 ^ 4, W / 2 ^ 4
                    P5: B, C_FPN, H / 2 ^ 5, W / 2 ^ 5
                    P6: B, C_FPN, H / 2 ^ 6, W / 2 ^ 6
                    P7: B, C_FPN, H / 2 ^ 7, W / 2 ^ 7
            Outputs:
                x: Tensor [B, (H3 * W3 * n_anchors
                               + H4 * W4 * n_anchors
                               + H5 * W5 * n_anchors
                               + H6 * W6 * n_anchors
                               + H7 * W7 * n_anchors), num_classes]
        '''
        features = []
        for x, batch_norm_layers in zip(pyramid_features, self.bn_list):
            for bn, conv in zip(batch_norm_layers, self.conv_list):
                x = self.swish(bn(conv(x)))

            x = self.header(x)
            x = x.permute(0, 2, 3, 1).contiguous()  # B x H x W x (num_anchors * num_classes)

            B, H, W, C = x.shape
            x = x.view(B, H, W, self.num_anchors, self.num_classes).contiguous()
            x = x.view(B, H * W * self.num_anchors, self.num_classes).contiguous()  # B x (H * W * num_anchors) x num_classes

            features.append(x)

        x = torch.cat(features, dim=1)
        x = x.sigmoid()

        return x
