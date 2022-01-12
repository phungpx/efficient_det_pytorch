import torch
from torch import nn
from torchvision.ops.boxes import batched_nms

from .bifpn import BiFPN
from .anchor import Anchors
from .head import Regressor, Classifier
from .transform import ClipBoxes, BBoxTransform
from .efficientnet.backbone import EfficientNetBackBone


class EfficientDetBackBone(nn.Module):
    def __init__(
        self,
        num_classes=80,
        compound_coef=0,
        backbone_pretrained=False,
        scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
        aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        iou_threshold=0.2,
        score_threshold=0.2
    ) -> None:
        super(EfficientDetBackBone, self).__init__()
        self.compound_coef = compound_coef
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        # self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

        self.aspect_ratios = aspect_ratios
        self.num_classes = num_classes
        self.num_scales = len(scales)

        # the channels of P3 / P4 / P5.
        conv_channel_coef = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(aspect_ratios) * len(scales)

        self.bifpn = nn.Sequential(
            *[BiFPN(num_channels=self.fpn_num_filters[self.compound_coef],
                    conv_channels=conv_channel_coef[compound_coef],
                    first_time=True if _ == 0 else False,
                    epsilon=1e-4, onnx_export=False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7) for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.regressor = Regressor(
            in_channels=self.fpn_num_filters[self.compound_coef],
            num_anchors=num_anchors,
            num_layers=self.box_class_repeats[self.compound_coef],
            pyramid_levels=self.pyramid_levels[self.compound_coef]
        )

        self.classifier = Classifier(
            in_channels=self.fpn_num_filters[self.compound_coef],
            num_anchors=num_anchors,
            num_classes=num_classes,
            num_layers=self.box_class_repeats[self.compound_coef],
            pyramid_levels=self.pyramid_levels[self.compound_coef]
        )

        self.anchors = Anchors(
            anchor_scale=self.anchor_scale[compound_coef],
            pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
            ratios=aspect_ratios,
            scales=scales
        )

        self.backbone_net = EfficientNetBackBone(
            compound_coef=self.backbone_compound_coef[compound_coef],
            load_weights=backbone_pretrained
        )

        # using for inference to find predicted bounding boxes
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes(compound_coef=compound_coef)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        anchors = self.anchors(inputs, inputs.dtype)
        regression = self.regressor(features)
        classification = self.classifier(features)

        return classification, regression, anchors

    def inference(self, inputs):
        predictions = []

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        anchors = self.anchors(inputs, inputs.dtype)
        regression = self.regressor(features)
        classification = self.classifier(features)

        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors)
        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > self.score_threshold)[:, :, 0]

        for i in range(inputs.shape[0]):
            if scores_over_thresh[i].sum() == 0:
                predictions.append(
                    {
                        'boxes': torch.FloatTensor([[0, 0, 1, 1]]),
                        'labels': torch.FloatTensor([-1]),
                        'scores': torch.FloatTensor([0])
                    }
                )
                continue

            classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
            transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            _scores, _classes = classification_per.max(dim=0)

            anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], _classes, iou_threshold=self.iou_threshold)

            if anchors_nms_idx.shape[0] != 0:
                _classes = _classes[anchors_nms_idx]
                _scores = _scores[anchors_nms_idx]
                _boxes = transformed_anchors_per[anchors_nms_idx, :]

                predictions.append({'boxes': _boxes, 'labels': _classes, 'scores': _scores})
            else:
                predictions.append(
                    {
                        'boxes': torch.FloatTensor([[0, 0, 1, 1]]),
                        'labels': torch.FloatTensor([-1]),
                        'scores': torch.FloatTensor([0])
                    }
                )

        return predictions


class EfficientDet(nn.Module):
    def __init__(
        self,
        pretrained_weight=None,
        head_only=False,
        num_classes=80,
        compound_coef=0,
        backbone_pretrained=False,
        scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
        aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        iou_threshold=0.2,
        score_threshold=0.2
    ) -> None:
        super(EfficientDet, self).__init__()
        self.model = EfficientDetBackBone(
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
        return self.model(inputs)

    def inference(self, inputs):
        return self.model.inference(inputs)
