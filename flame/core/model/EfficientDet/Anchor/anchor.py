import torch
import itertools
from torch import nn
from typing import List, Tuple


class AnchorGeneration(nn.Module):
    def __init__(self,
                 compound_coef: int = 0,
                 scales: List[float] = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
                 aspect_ratios: List[Tuple[float, float]] = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]) -> None:
        super(AnchorGeneration, self).__init__()
        self.anchor_scale = 4. if compound_coef != 7 else 5.

        self.scales = scales
        self.aspect_ratios = aspect_ratios

    def forward(self, inputs: torch.Tensor, pyramid_features: Tuple[torch.Tensor]) -> torch.Tensor:
        image_size = inputs.shape[-2:]
        grid_sizes = [pyramid_feature.shape[-2:] for pyramid_feature in pyramid_features]
        dtype, device = pyramid_features[0].dtype, pyramid_features[0].device
        strides = [[image_size[0] // grid_size[0], image_size[1] // grid_size[1]] for grid_size in grid_sizes]

        anchors_over_all_pyramid_features = []
        for stride in strides:
            stride_height, stride_width = stride

            anchors_per_pyramid_feature = []
            for scale, aspect_ratio in itertools.product(self.scales, self.aspect_ratios):
                if (image_size[0] % stride_height != 0) or (image_size[1] % stride_width != 0):
                    raise ValueError('input size must be divided by the stride.')

                base_anchor_width = self.anchor_scale * stride_width * scale
                base_anchor_height = self.anchor_scale * stride_height * scale

                anchor_size_center_x = base_anchor_width * aspect_ratio[0] / 2.0
                anchor_size_center_y = base_anchor_height * aspect_ratio[1] / 2.0

                shift_x = torch.arange(
                    start=stride_width / 2, end=image_size[1], step=stride_width,
                    dtype=torch.float32, device=device
                )
                shift_y = torch.arange(
                    start=stride_height / 2, end=image_size[0], step=stride_height,
                    dtype=torch.float32, device=device
                )

                shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
                shift_y, shift_x = shift_x.reshape(-1), shift_y.reshape(-1)

                # x1, y1, x2, y2
                boxes = torch.stack((shift_x - anchor_size_center_x,
                                     shift_y - anchor_size_center_y,
                                     shift_x + anchor_size_center_x,
                                     shift_y + anchor_size_center_y), dim=1)
                anchors_per_pyramid_feature.append(boxes)
            anchors_per_pyramid_feature = torch.cat(anchors_per_pyramid_feature, dim=0)
            anchors_over_all_pyramid_features.append(anchors_per_pyramid_feature)

        anchor_boxes = torch.cat(anchors_over_all_pyramid_features, dim=0).to(dtype).to(device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        return anchor_boxes
