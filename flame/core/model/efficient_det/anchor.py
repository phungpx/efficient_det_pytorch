import math
import torch
import itertools
import numpy as np
import torch.nn as nn

from typing import List, Tuple


class Anchors(nn.Module):
    def __init__(
        self,
        anchor_scale: float = 4.,  # NOTE!!: anchor_scale = 4. if compound_coef != 7 else 5.
        scales: List[float] = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
        aspect_ratios: List[float] = [0.5, 1., 2.]  # width_box / height_box
    ):
        super(Anchors, self).__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale

    def forward(self, inputs: torch.Tensor, features: Tuple[torch.Tensor]) -> torch.Tensor:
        """Generates multiscale anchor boxes.
        Args:
            inputs: Tensor (B x N x H x W): H = W = 128*compound_coef + 512
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
