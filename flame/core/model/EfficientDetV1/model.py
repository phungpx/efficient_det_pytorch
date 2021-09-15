import torch
from torch import nn
from typing import Optional, List, Tuple
from .efficientdet import EfficientDet


class Model(nn.Module):
    def __init__(self,
                 num_classes: int = 80,
                 compound_coef: int = 0,
                 head_only: bool = False,
                 model_weight_path: Optional[str] = None,
                 backbone_weight_path: Optional[str] = None,
                 backbone_pretrained_weight: bool = False,
                 scales: List[float] = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
                 aspect_ratios: List[Tuple[float, float]] = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                 iou_threshold: float = 0.2, score_threshold: float = 0.2) -> None:
        super(Model, self).__init__()
        self.model = EfficientDet(
            num_classes=num_classes,
            compound_coef=compound_coef,
            backbone_weight_path=backbone_weight_path,
            backbone_pretrained_weight=backbone_pretrained_weight,
            scales=scales,
            aspect_ratios=aspect_ratios,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold
        )

        if model_weight_path is not None:
            state_dict = torch.load(model_weight_path, map_location='cpu')
            state_dict.pop('classifier.header.pointwise_conv.conv.weight')
            state_dict.pop('classifier.header.pointwise_conv.conv.bias')
            self.model.load_state_dict(state_dict, strict=False)

        if head_only:
            self.model.apply(self.freeze_backbone)

    def freeze_backbone(self, m):
        classname = m.__class__.__name__
        for ntl in ['EfficientNet', 'BiFPN']:
            if ntl in classname:
                for param in m.parameters():
                    param.requires_grad = False

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, inputs):
        return self.model._forward(inputs)

    def inference(self, inputs):
        return self.model._detect(inputs)
