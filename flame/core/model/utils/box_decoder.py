import torch
from torch import nn
from typing import Tuple


class BoxDecoder(nn.Module):
    def __init__(
        self,
        mean: Tuple[float, float, float, float] = (0, 0, 0, 0),  # x1, y1, x2, y2 respectively
        std: Tuple[float, float, float, float] = (1, 1, 1, 1),  # x1, y1, x2, y2 respectively
    ):
        super(BoxDecoder, self).__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def forward(self, anchors: torch.Tensor, regression: torch.Tensor) -> torch.Tensor:
        """decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py
            * (x, y): center of ground truth box
            * (xa, ya): center of anchor box
            - dx = (x - xa) / wa --> x = dx * wa + xa --> x1 = x - w / 2
            - dy = (y - ya) / ha --> y = dy * ha + ya --> y1 = y - h / 2
            - dw = log(w / wa)   --> w = e ^ dw * wa  --> x2 = x + w / 2
            - dh = log(h / ha)   --> h = e ^ dh * ha  --> y2 = y + h / 2
            Args:
                anchors: [1, num_boxes, 4] (y1, x1, y2, x2)
                         -> xa, ya = (x2 + x1) / 2, (y2 + y1) / 2
                         -> wa, ha = x2 - x1, y2 - y1
                regression: [batch_size, num_boxes, 4] (dy, dx, dh, dw)
            Returns:
                bounding_boxes: [batch_size, num_boxes, 4] (x1, y1, x2, y2)
        """
        # to device
        self.mean.to(anchors.device)
        self.std.to(anchors.device)

        xa = (anchors[:, :, 3] + anchors[:, :, 1]) / 2.
        ya = (anchors[:, :, 2] + anchors[:, :, 0]) / 2.
        wa = torch.clamp(anchors[:, :, 3] - anchors[:, :, 1], min=1.)
        ha = torch.clamp(anchors[:, :, 2] - anchors[:, :, 0], min=1.)

        # denormalized regression from x' = (x - mean) / std -> x = x' * std + mean
        dx = regression[:, :, 1] * self.std[1] + self.mean[1]
        dy = regression[:, :, 0] * self.std[0] + self.mean[0]
        dw = regression[:, :, 3] * self.std[3] + self.mean[3]
        dh = regression[:, :, 2] * self.std[2] + self.mean[2]

        x = xa + dx * wa
        y = ya + dy * ha
        w = dw.exp() * wa
        h = dh.exp() * ha

        x1 = x - w / 2.
        y1 = y - h / 2.
        x2 = x + w / 2.
        y2 = y + h / 2.

        return torch.stack([x1, y1, x2, y2], dim=2)

# class BBoxTransform(nn.Module):
#     def forward(self, anchors: torch.Tensor, regression: torch.Tensor) -> torch.Tensor:
#         """decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py
#         * (x, y): center of ground truth box
#         * (xa, ya): center of anchor box
#         - dx = (x - xa) / wa --> x = dx * wa + xa --> x1 = x - w / 2
#         - dy = (y - ya) / ha --> y = dy * ha + ya --> y1 = y - h / 2
#         - dw = log(w / wa)   --> w = e ^ dw * wa  --> x2 = x + w / 2
#         - dh = log(h / ha)   --> h = e ^ dh * ha  --> y2 = y + h / 2
#         Args:
#             anchors: [1, num_boxes, 4] (y1, x1, y2, x2)
#                      -> xa, ya = (x2 + x1) / 2, (y2 + y1) / 2
#                      -> wa, ha = x2 - x1, y2 - y1
#             regression: [batch_size, num_boxes, 4] (dy, dx, dh, dw)
#         Returns:
#             bounding_boxes: [batch_size, num_boxes, 4] (x1, y1, x2, y2)
#         """
#         xa = (anchors[:, :, 3] + anchors[:, :, 1]) / 2.
#         ya = (anchors[:, :, 2] + anchors[:, :, 0]) / 2.
#         wa = torch.clamp(anchors[:, :, 3] - anchors[:, :, 1], min=1.)
#         ha = torch.clamp(anchors[:, :, 2] - anchors[:, :, 0], min=1.)

#         x = regression[:, :, 1] * wa + xa
#         y = regression[:, :, 0] * ha + ya
#         w = regression[:, :, 3].exp() * wa
#         h = regression[:, :, 2].exp() * ha

#         x1 = x - w / 2.
#         y1 = y - h / 2.
#         x2 = x + w / 2.
#         y2 = y + h / 2.

#         return torch.stack([x1, y1, x2, y2], dim=2)
