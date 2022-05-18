import torch
from torch import nn


class BoxClipper(nn.Module):
    def __init__(self):
        super(BoxClipper, self).__init__()

    def forward(self, boxes: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=image_width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=image_height - 1)

        return boxes
