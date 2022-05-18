import math
import torch
import itertools
import numpy as np
import torch.nn as nn

from typing import List, Tuple


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        anchor_scale: float = 4.,  # for P1, P2 (each layer downsize 2 times) -> 2 ** 2
        scales: List[float] = [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)],
        aspect_ratios: List[float] = [0.5, 1., 2.]  # width_box / height_box
    ):
        super(AnchorGenerator, self).__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale

    def forward(self, inputs: torch.Tensor, features: Tuple[torch.Tensor]) -> torch.Tensor:
        """Generates multiscale anchor boxes.
        Args:
            inputs: Tensor (B x N x H x W): H = W = 128 * compound_coef + 512
            features: Tuple (Tensor[B x N' x H' x W']): tuple of tensors get from output of biFPN
        Output:
            anchors: Tensor[1 x all_anchors x 4]: all anchors of all pyramid features
        """

        dtype, device = inputs.dtype, inputs.device
        _, _, image_height, image_width = inputs.shape   # inputs: B x N x H x W

        # stride of anchors on input size
        features_sizes = [feature.shape[2:] for feature in features]   # List[[H_feature, W_feature]]
        strides = [
            (image_height // feature_height, image_width // feature_width)
            for feature_height, feature_width in features_sizes
        ]

        anchors_over_all_pyramid_features = []
        for stride_height, stride_width in strides:

            anchors_per_pyramid_feature = []
            for scale, ratio in itertools.product(self.scales, self.aspect_ratios):
                if (image_width % stride_width != 0) or (image_height % stride_height != 0):
                    raise ValueError('input size must be divided by the stride.')

                # anchor base size
                base_anchor_width = self.anchor_scale * stride_width
                base_anchor_height = self.anchor_scale * stride_height

                # anchor size
                anchor_width = base_anchor_width * scale * math.sqrt(ratio)
                anchor_height = base_anchor_height * scale * math.sqrt(1 / ratio)

                # center of anchors
                cx = torch.arange(
                    start=stride_width / 2, end=image_width, step=stride_width, device=device, dtype=dtype
                )
                cy = torch.arange(
                    start=stride_height / 2, end=image_height, step=stride_height, device=device, dtype=dtype
                )

                cx, cy = torch.meshgrid(cx, cy)
                cx, cy = cx.t().reshape(-1), cy.t().reshape(-1)

                # coodinates of each anchors: format anchor boxes # y1,x1,y2,x2
                anchors = torch.stack(
                    (
                        cy - anchor_height / 2., cx - anchor_width / 2.,
                        cy + anchor_height / 2., cx + anchor_width / 2.,
                    ), dim=1
                )  # num_anchors x 4

                anchors = anchors.unsqueeze(dim=1)  # num_anchors x 1 x 4
                anchors_per_pyramid_feature.append(anchors)

            # num_anchors x (scale * aspect_ratios) x 4
            anchors_per_pyramid_feature = torch.cat(anchors_per_pyramid_feature, dim=1)
            # (num_anchors * scale * aspect_ratios) x 4
            anchors_per_pyramid_feature = anchors_per_pyramid_feature.reshape(-1, 4)
            anchors_over_all_pyramid_features.append(anchors_per_pyramid_feature)

        # [(num_anchors * scale * aspect_ratios) * pyramid_levels] x 4
        anchors = torch.vstack(anchors_over_all_pyramid_features)

        return anchors.unsqueeze(dim=0)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = torch.rand(size=[1, 3, 512, 512], dtype=torch.float32, device=device)
    features = (
        torch.rand(size=[1, 256, 64, 64], dtype=torch.float32, device=device),  # P3
        torch.rand(size=[1, 256, 32, 32], dtype=torch.float32, device=device),  # P4
        torch.rand(size=[1, 256, 16, 16], dtype=torch.float32, device=device),  # P5
        torch.rand(size=[1, 256, 8, 8], dtype=torch.float32, device=device),  # P6
        torch.rand(size=[1, 256, 4, 4], dtype=torch.float32, device=device),  # P7
    )

    anchor_generator = AnchorGenerator()

    anchors = anchor_generator(inputs=inputs, features=features)

    print(anchors.shape)  # torch.Size([1, 49104, 4])
