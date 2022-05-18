import torch
from torch import nn
from torchvision.ops.boxes import batched_nms
from typing import List, Dict

from .bifpn import BiFPN
from .backbone import load_backbone
from .head import Regressor, Classifier
from .anchor_generator import AnchorGenerator

from .utils.box_clipper import BoxClipper
from .utils.box_decoder import BoxDecoder


BACKBONE_NAME = {
    0: 'efficientnet-b0',
    1: 'efficientnet-b1',
    2: 'efficientnet-b2',
    3: 'efficientnet-b3',
    4: 'efficientnet-b4',
    5: 'efficientnet-b5',
    6: 'efficientnet-b6',
    7: 'efficientnet-b6',
    8: 'efficientnet-b7'
}


class EfficientDet(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        compound_coef: int = 0,
        backbone_pretrained: bool = False,
        scales: List[float] = [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)],
        aspect_ratios: List[float] = [0.5, 1., 2.],
        iou_threshold: float = 0.2,
        score_threshold: float = 0.2
    ) -> None:
        super(EfficientDet, self).__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        NUMBER_PYRAMID_LEVELS = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        NUMBER_BiFPN_BLOCKS = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        NUMBER_CONV_HEAD_BLOCKS = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        BiFPN_CHANNELS = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        # self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

        # backbone
        self.backbone_net, backbone_out_channels = load_backbone(
            backbone_name=BACKBONE_NAME[compound_coef],
            pretrained=backbone_pretrained
        )

        # neck
        self.bifpn = nn.Sequential(
            *[
                BiFPN(
                    BiFPN_out_channels=BiFPN_CHANNELS[compound_coef],
                    P3_out_channels=backbone_out_channels['C3'],
                    P4_out_channels=backbone_out_channels['C4'],
                    P5_out_channels=backbone_out_channels['C5'],
                    first_time=True if i == 0 else False,
                    epsilon=1e-4, onnx_export=False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7
                )
                for i in range(NUMBER_BiFPN_BLOCKS[compound_coef])
            ]
        )

        # head
        self.regressor = Regressor(
            BiFPN_out_channels=BiFPN_CHANNELS[compound_coef],
            num_anchors=len(scales) * len(aspect_ratios),
            num_layers=NUMBER_CONV_HEAD_BLOCKS[compound_coef],
            num_pyramid_levels=NUMBER_PYRAMID_LEVELS[compound_coef]
        )

        self.classifier = Classifier(
            num_classes=num_classes,
            BiFPN_out_channels=BiFPN_CHANNELS[compound_coef],
            num_anchors=len(scales) * len(aspect_ratios),
            num_layers=NUMBER_CONV_HEAD_BLOCKS[compound_coef],
            num_pyramid_levels=NUMBER_PYRAMID_LEVELS[compound_coef]
        )

        # anchor
        self.anchor_generator = AnchorGenerator(aspect_ratios=aspect_ratios,scales=scales)

        # using for post processing to find candidate boxes
        self.box_clipper = BoxClipper()
        self.box_decoder = BoxDecoder()

    @property
    def freeze_batchnorm(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, inputs: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        _, _, P3, P4, P5 = self.backbone_net(inputs)
        pyramid_features = self.bifpn((P3, P4, P5))
        reg_preds = self.regressor(pyramid_features)  # B x all_anchors x 4
        cls_preds = self.classifier(pyramid_features)  # B x all_anchors x num_classes

        anchors = self.anchor_generator(inputs, pyramid_features)  # 1 x all_anchors x 4

        return cls_preds, reg_preds, anchors

    def predict(self, inputs: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        _, _, P3, P4, P5 = self.backbone_net(inputs)
        pyramid_features = self.bifpn((P3, P4, P5))
        reg_preds = self.regressor(pyramid_features)  # B x all_anchors x 4
        cls_preds = self.classifier(pyramid_features)  # B x all_anchors x num_classes

        anchors = self.anchor_generator(inputs, pyramid_features)  # 1 x all_anchors x 4

        # get predicted boxes which are decoded from reg_preds and anchors
        batch_boxes = self.box_decoder(anchors=anchors, regression=reg_preds)  # B x all_anchors x 4, decoded boxes
        batch_boxes = self.box_clipper(boxes=batch_boxes, image_width=inputs.shape[3], image_height=inputs.shape[2])

        batch_scores, batch_classes = torch.max(cls_preds, dim=2)  # B x all_anchors, choose class with max confidence in each box.
        batch_scores_over_threshold = (batch_scores > self.score_threshold)  # B x all_anchors, remove class of box with confidence lower than threshold.

        preds = []
        batch_size = inputs.shape[0]
        for i in range(batch_size):  # loop for each sample.
            sample_scores_over_threshold = batch_scores_over_threshold[i]  # all_anchors, sample_scores_over_threshold
            if sample_scores_over_threshold.sum() == 0:  # sample has no valid boxes.
                preds.append(
                    {
                        'boxes': torch.FloatTensor([[0, 0, 1, 1]]),  # 1 pixel.
                        'labels': torch.FloatTensor([-1]),
                        'scores': torch.FloatTensor([0])
                    }
                )
                continue

            sample_boxes = batch_boxes[i]  # all_anchors x 4
            sample_scores = batch_scores[i]  # all_anchors
            sample_classes = batch_classes[i]  # all_anchors

            valid_boxes = sample_boxes[sample_scores_over_threshold, :]  # n_valid_scores x 4
            valid_scores = sample_scores[sample_scores_over_threshold]  # n_valid_scores
            valid_classes = sample_classes[sample_scores_over_threshold]  # n_valid_scores

            # determind what boxes will be kept by nms algorithm
            keep_indices = batched_nms(
                boxes=valid_boxes,
                scores=valid_scores,
                idxs=valid_classes,
                iou_threshold=self.iou_threshold
            )

            if keep_indices.shape[0] != 0:
                kept_boxes = valid_boxes[keep_indices, :]  # num_keep_boxes x 4
                kept_scores = valid_scores[keep_indices]  # num_keep_boxes
                kept_classes = valid_classes[keep_indices]  # num_keep_boxes

                preds.append({'boxes': kept_boxes, 'labels': kept_classes, 'scores': kept_scores})
            else:
                preds.append(
                    {
                        'boxes': torch.FloatTensor([[0, 0, 1, 1]]),
                        'labels': torch.FloatTensor([-1]),
                        'scores': torch.FloatTensor([0])
                    }
                )

        return preds


class Model(nn.Module):
    def __init__(
        self,
        pretrained_weight: str = None,
        # head_only: bool = False,
        num_classes: int = 80,
        compound_coef: int = 0,
        backbone_pretrained: bool = False,
        scales: List[float] = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
        aspect_ratios: List[float] = [0.5, 1.0, 2.0],
        iou_threshold: float = 0.2,
        score_threshold: float = 0.2
    ) -> None:
        super(Model, self).__init__()
        self.model = EfficientDet(
            num_classes=num_classes,
            compound_coef=compound_coef,
            backbone_pretrained=backbone_pretrained,
            scales=scales,
            aspect_ratios=aspect_ratios,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )

        if pretrained_weight is not None:
            state_dict = torch.load(pretrained_weight, map_location='cpu')
            state_dict.pop('classifier.header.pointwise_conv.conv.weight')
            state_dict.pop('classifier.header.pointwise_conv.conv.bias')
            state_dict.pop('regressor.header.pointwise_conv.conv.weight')
            state_dict.pop('regressor.header.pointwise_conv.conv.bias')
            self.model.load_state_dict(state_dict, strict=False)

    #     if head_only:
    #         self.model.freeze_backbone

    # def freeze_backbone(self, m):
    #     classname = m.__class__.__name__
    #     for ntl in ['EfficientNet', 'BiFPN']:
    #         if ntl in classname:
    #             for param in m.parameters():
    #                 param.requires_grad = False

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, inputs):
        return self.model(inputs)

    def predict(self, inputs):
        return self.model.predict(inputs)
