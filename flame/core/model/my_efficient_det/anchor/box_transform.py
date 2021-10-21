import torch
from torch import nn


__all__ = ['BoxClipper', 'BoxDecoder']


class BoxClipper(nn.Module):
    def __init__(self, compound_coef: int = 0):
        super(BoxClipper, self).__init__()
        self.compound_coef = compound_coef

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        image_height = image_width = 512 + 128 * self.compound_coef

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=image_width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=image_height - 1)

        return boxes


class BoxDecoder(nn.Module):
    def __init__(self):
        super(BoxDecoder, self).__init__()

    def forward(self, anchors: torch.Tensor, regression: torch.Tensor) -> torch.Tensor:
        """decode_box_outputs adapted from
        https://github.com/google/automl/blob/master/efficientdet/anchors.py
        Args:
            anchors: [1, num_anchors, (y1, x1, y2, x2)]
            regression: [batch_size, num_boxes, (dy, dx, dh, dw)]
        Outputs:

        """
        cy_a = (anchors[..., 0] + anchors[..., 2]) / 2
        cx_a = (anchors[..., 1] + anchors[..., 3]) / 2
        h_a = anchors[..., 2] - anchors[..., 0]
        w_a = anchors[..., 3] - anchors[..., 1]

        w_b = regression[..., 3].exp() * w_a
        h_b = regression[..., 2].exp() * h_a

        cy_b = regression[..., 0] * h_a + cy_a
        cx_b = regression[..., 1] * w_a + cx_a

        ymin = cy_b - h_b / 2.
        xmin = cx_b - w_b / 2.
        ymax = cy_b + h_b / 2.
        xmax = cx_b + w_b / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)
