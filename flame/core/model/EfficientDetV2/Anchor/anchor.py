import torch
import itertools
import numpy as np
from torch import nn
from typing import List, Tuple


class AnchorGeneration(nn.Module):
    def __init__(self,
                 debug: bool = False,
                 compound_coef: int = 0,
                 scales: List[float] = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
                 aspect_ratios: List[float] = [0.5, 1., 2.]) -> None:
        super(AnchorGeneration, self).__init__()
        self.anchor_scale = 4. if compound_coef != 7 else 5.

        self.debug = debug
        self.scales = scales
        self.aspect_ratios = aspect_ratios

    def forward(self, inputs: torch.Tensor, pyramid_features: Tuple[torch.Tensor]) -> torch.Tensor:
        image_size = inputs.shape[-2:]
        grid_sizes = [pyramid_feature.shape[-2:] for pyramid_feature in pyramid_features]
        dtype, device = pyramid_features[0].dtype, pyramid_features[0].device
        strides = [[image_size[0] // grid_size[0], image_size[1] // grid_size[1]] for grid_size in grid_sizes]

        if self.debug:
            visual_image = np.zeros(shape=(image_size[0], image_size[1], 3), dtype=np.uint8)

        anchors_over_all_pyramid_features = []
        for stride in strides:
            stride_height, stride_width = stride

            anchors_per_pyramid_feature = []
            for scale, aspect_ratio in itertools.product(self.scales, self.aspect_ratios):
                if (image_size[0] % stride_height != 0) or (image_size[1] % stride_width != 0):
                    raise ValueError('input size must be divided by the stride.')

                base_anchor_width = self.anchor_scale * stride_width * scale
                base_anchor_height = self.anchor_scale * stride_height * scale

                anchor_width = base_anchor_width * np.sqrt(aspect_ratio)
                anchor_height = base_anchor_height * (1 / np.sqrt(aspect_ratio))

                shift_x = torch.arange(
                    start=stride_width / 2, end=image_size[1], step=stride_width,
                    dtype=torch.float32, device=device
                )
                shift_y = torch.arange(
                    start=stride_height / 2, end=image_size[0], step=stride_height,
                    dtype=torch.float32, device=device
                )

                shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
                shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

                # y1, x1, y2, x2
                anchors = torch.stack(
                    (shift_y - anchor_height / 2.,
                     shift_x - anchor_width / 2.,
                     shift_y + anchor_height / 2.,
                     shift_x + anchor_width / 2.),
                    dim=1
                )

                anchors_per_pyramid_feature.append(anchors)

                if self.debug:
                    import cv2
                    for anchor in anchors:
                        y1, x1, y2, x2 = anchor.numpy()
                        cv2.rectangle(
                            img=visual_image,
                            pt1=(int(round(x1)), int(round(y1))),
                            pt2=(int(round(x2)), int(round(y2))),
                            color=(255, 255, 255),
                            thickness=1
                        )
                    cv2.imshow(f'visual_at_stride_#{stride}', visual_image)
                    cv2.waitKey()
                    cv2.destroyAllWindows()

            anchors_per_pyramid_feature = torch.cat(anchors_per_pyramid_feature, dim=0)

            anchors_over_all_pyramid_features.append(anchors_per_pyramid_feature)

        anchor_boxes = torch.cat(anchors_over_all_pyramid_features, dim=0).to(dtype).to(device)

        return anchor_boxes.unsqueeze(0)


if __name__ == "__main__":
    anchor_generator = AnchorGeneration(debug=True,
                                        compound_coef=0,
                                        scales=[1 / 16, 1 / 8, 1 / 4],
                                        aspect_ratios=[1 / 3, 0.5, 1., 2., 3.])

    inputs = torch.rand(1, 3, 512, 512)
    pyramid_features = [torch.rand(1, 3, 4, 4)]

    anchor_boxes = anchor_generator(inputs, pyramid_features)
