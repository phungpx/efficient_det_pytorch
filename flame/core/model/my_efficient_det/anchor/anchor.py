import torch
import itertools
import numpy as np
from torch import nn
from typing import List, Tuple


class AnchorGeneration(nn.Module):
    def __init__(
        self,
        compound_coef: int = 0,
        scales: List[float] = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
        aspect_ratios: List[float] = [0.5, 1., 2.]
    ) -> None:
        super(AnchorGeneration, self).__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = 4. if compound_coef != 7 else 5.

    def forward(
        self,
        inputs: torch.Tensor,
        pyramid_features: Tuple[torch.Tensor]
    ) -> torch.Tensor:

        dtype, device = pyramid_features[0].dtype, pyramid_features[0].device

        image_h, image_w = inputs.shape[-2:]
        map_sizes = [pyramid_feature.shape[-2:] for pyramid_feature in pyramid_features]
        strides = [(image_h // map_h, image_w // map_w) for map_h, map_w in map_sizes]

        anchors_over_all_pyramid_features = []
        for stride in strides:
            stride_h, stride_w = stride

            anchors_per_pyramid_feature = []
            for scale, aspect_ratio in itertools.product(self.scales, self.aspect_ratios):
                if (image_h % stride_h != 0) or (image_w % stride_w != 0):
                    raise ValueError('input size must be divided by the stride.')

                base_anchor_w = self.anchor_scale * stride_w * scale
                base_anchor_h = self.anchor_scale * stride_h * scale

                anchor_w = base_anchor_w * np.sqrt(aspect_ratio)
                anchor_h = base_anchor_h * (1 / np.sqrt(aspect_ratio))

                cx = torch.arange(
                    start=stride_w / 2,
                    end=image_w,
                    step=stride_w,
                    dtype=torch.float32,
                    device=device,
                )

                cy = torch.arange(
                    start=stride_h / 2,
                    end=image_h,
                    step=stride_h,
                    dtype=torch.float32,
                    device=device,
                )

                cx, cy = torch.meshgrid(cx, cy)
                cx, cy = cx.reshape(-1), cy.reshape(-1)

                # y1, x1, y2, x2
                anchors = torch.stack(
                    (
                        cy - anchor_h / 2.,
                        cx - anchor_w / 2.,
                        cy + anchor_h / 2.,
                        cx + anchor_w / 2.
                    ),
                    dim=1
                )

                anchors_per_pyramid_feature.append(anchors)

            anchors_per_pyramid_feature = torch.cat(anchors_per_pyramid_feature, dim=0)

            anchors_over_all_pyramid_features.append(anchors_per_pyramid_feature)

        anchor_boxes = torch.cat(anchors_over_all_pyramid_features, dim=0).to(dtype).to(device)

        return anchor_boxes.unsqueeze(0)
