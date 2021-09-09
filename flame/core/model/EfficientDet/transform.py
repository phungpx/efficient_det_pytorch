import torch
import torch.nn as nn


__all__ = ['BBoxTransform', 'ClipBoxes']


class BBoxTransform(nn.Module):
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
        xa = (anchors[:, :, 3] + anchors[:, :, 1]) / 2.
        ya = (anchors[:, :, 2] + anchors[:, :, 0]) / 2.
        wa = torch.clamp(anchors[:, :, 3] - anchors[:, :, 1], min=1.)
        ha = torch.clamp(anchors[:, :, 2] - anchors[:, :, 0], min=1.)

        x = regression[:, :, 1] * wa + xa
        y = regression[:, :, 0] * ha + ya
        w = regression[:, :, 3].exp() * wa
        h = regression[:, :, 2].exp() * ha

        x1 = x - w / 2.
        y1 = y - h / 2.
        x2 = x + w / 2.
        y2 = y + h / 2.

        return torch.stack([x1, y1, x2, y2], dim=2)


class ClipBoxes(nn.Module):
    def __init__(self, compound_coef):
        super(ClipBoxes, self).__init__()
        self.compound_coef = compound_coef

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        image_height = image_width = 512 + 128 * self.compound_coef

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=image_width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=image_height - 1)

        return boxes
