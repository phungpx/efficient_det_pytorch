from .bi_fpn.bifpn import BiFPN
from .head.regressor import Regressor
from .head.classifier import Classifier
from .anchor.anchor import AnchorGeneration
from .anchor.box_transform import BoxClipper, BoxDecoder
from .efficient_net.back_bone import EfficientNetBackBone

import torch
from torch import nn
from typing import List, Tuple, Dict, Optional
from torchvision.ops.boxes import batched_nms


class EfficientDet(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        compound_coef: int = 0,
        backbone_weight_path: Optional[str] = None,
        backbone_pretrained_weight: bool = False,
        scales: List[float] = None,
        aspect_ratios: List[float] = None,
        score_threshold: float = 0.2,
        iou_threshold: float = 0.2
    ) -> None:
        super(EfficientDet, self).__init__()

        # input image resolution: R_input = 512 + 128 * compound_coef
        self.R_input = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

        # out channels of each BiFPN Layer: W_bifpn = 64 * (1.35) ** compound_coef (#channels)
        self.W_bifpn = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        # the number of BiFPN Layers: D_bifpn = 3 + compound_coef (#layers)
        self.D_bifpn = [3, 4, 5, 6, 7, 7, 8, 8, 8]

        # in / out channels of each Conv Layer in Head: W_pred = W_bifpn (#channels)
        self.W_pred = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        # the number of Regression Head Layers: D_class = D_box = 3 + [compound_coef / 3] (#layers)
        self.D_box = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        # the number of Classification Head Layers: D_class = D_box = 3 + [compound_coef / 3] (#layers)
        self.D_class = [3, 3, 3, 4, 4, 4, 5, 5, 5]

        # out_channels of P3, P4, P5 after feature extractor class
        self.backbone_out_channels = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640]
        }

        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.feature_extractor = EfficientNetBackBone(
            R_input=self.R_input,
            compound_coef=compound_coef,
            pretrained_weight=backbone_pretrained_weight,
            weight_path=backbone_weight_path
        )

        self.bifpn = BiFPN(
            compound_coef=compound_coef,
            backbone_out_channels=self.backbone_out_channels,
            W_bifpn=self.W_bifpn,
            D_bifpn=self.D_bifpn,
            onnx_export=False
        )

        num_anchors = len(scales) * len(aspect_ratios)

        self.classifier = Classifier(
            n_classes=num_classes,
            n_anchors=num_anchors,
            compound_coef=compound_coef,
            D_class=self.D_class,
            W_pred=self.W_pred,
            onnx_export=False
        )

        self.regressor = Regressor(
            n_anchors=num_anchors,
            compound_coef=compound_coef,
            D_box=self.D_box,
            W_pred=self.W_pred,
            onnx_export=False
        )

        self.anchor_generator = AnchorGeneration(
            compound_coef=compound_coef,
            scales=scales,
            aspect_ratios=aspect_ratios
        )

        self.box_decoder = BoxDecoder()
        self.box_clipper = BoxClipper(compound_coef=compound_coef)

    def _detect(self, inputs: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        predictions = []

        cls_preds, loc_preds, anchor_boxes = self._forward(inputs=inputs)

        transformed_anchors = self.box_decoder(anchor_boxes, loc_preds)
        transformed_anchors = self.box_clipper(transformed_anchors, inputs)

        scores = torch.max(cls_preds, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > self.score_threshold)[:, :, 0]

        for i in range(inputs.shape[0]):
            if scores_over_thresh[i].sum() == 0:
                predictions.append(
                    {
                        'boxes': torch.tensor([[0, 0, 1, 1]], dtype=torch.float32),
                        'labels': torch.tensor([-1], dtype=torch.int64),
                        'scores': torch.tensor([0], dtype=torch.float32),
                    }
                )
                continue

            classification_per = cls_preds[i, scores_over_thresh[i, :], ...].permute(1, 0)
            transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            _scores, _classes = classification_per.max(dim=0)

            anchors_nms_idx = batched_nms(
                transformed_anchors_per,
                scores_per[:, 0],
                _classes,
                iou_threshold=self.iou_threshold)

            if anchors_nms_idx.shape[0] != 0:
                _classes = _classes[anchors_nms_idx]
                _scores = _scores[anchors_nms_idx]
                _boxes = transformed_anchors_per[anchors_nms_idx, :]

                predictions.append(
                    {
                        'boxes': _boxes,
                        'labels': _classes,
                        'scores': _scores
                    }
                )
            else:
                predictions.append(
                    {
                        'boxes': torch.tensor([[0, 0, 1, 1]], dtype=torch.float32),
                        'labels': torch.tensor([-1], dtype=torch.int64),
                        'scores': torch.tensor([0], dtype=torch.float32),
                    }
                )

        return predictions

    def _forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Args:
            inputs: B x C x H x W
        Outputs:
            ** num_anchors: = H3 * W3 * n_anchors + H4 * W4 * n_anchors
                            + H5 * W5 * n_anchors + H6 * W6 * n_anchors
                            + H7 * W7 * n_anchors
            cls_preds: Tensor [B, num_anchors, n_classes]
            loc_preds: Tensor [B, num_anchors, 4]
            anchors: Tensor [1, num_anchors, 4]
        '''
        feature_maps = self.feature_extractor(x=inputs)  # P1, P2, P3, P4, P5

        P3, P4, P5 = feature_maps[-3:]
        pyramid_features = self.bifpn(feature_maps=(P3, P4, P5))

        anchors = self.anchor_generator(inputs=inputs, pyramid_features=pyramid_features)
        # anchors = self.anchor_generator(inputs, inputs.dtype)

        cls_preds = self.classifier(pyramid_features=pyramid_features)
        loc_preds = self.regressor(pyramid_features=pyramid_features)

        return cls_preds, loc_preds, anchors