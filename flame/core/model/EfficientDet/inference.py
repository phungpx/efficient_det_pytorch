from .Anchor.transform import ClipBoxes, BBoxTransform

import torch
from torch import nn
from typing import List, Dict
from torchvision.ops.boxes import batched_nms


class Inference(nn.Module):
    def __init__(self,
                 compound_coef: int = 0,
                 score_threshold: float = 0.2,
                 iou_threshold: float = 0.2) -> None:
        super(Inference, self).__init__()

        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        self.bbox_decoder = BBoxTransform()
        self.bbox_clipper = ClipBoxes(compound_coef=compound_coef)

    def _predict(self, cls_preds: torch.Tensor, loc_preds: torch.Tensor,
                 anchors: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        predictions = []

        regressed_bboxes = self.bbox_decoder(anchors, loc_preds)
        regressed_bboxes = self.bbox_clipper(regressed_bboxes)

        scores = torch.max(cls_preds, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > self.score_threshold)[:, :, 0]

        for i in range(cls_preds.shape[0]):
            if scores_over_thresh[i].sum() == 0:
                predictions.append({'boxes': torch.FloatTensor([[0, 0, 1, 1]]),
                                    'labels': torch.FloatTensor([-1]),
                                    'scores': torch.FloatTensor([0])})
                continue

            cls_pred_per_sample = cls_preds[i, scores_over_thresh[i, :], ...].permute(1, 0)
            bbox_pred_per_sample = regressed_bboxes[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            _scores, _classes = cls_pred_per_sample.max(dim=0)

            anchors_nms_idx = batched_nms(bbox_pred_per_sample, scores_per[:, 0], _classes,
                                          iou_threshold=self.iou_threshold)

            if anchors_nms_idx.shape[0] != 0:
                _classes = _classes[anchors_nms_idx]
                _scores = _scores[anchors_nms_idx]
                _boxes = bbox_pred_per_sample[anchors_nms_idx, :]

                predictions.append({'boxes': _boxes, 'labels': _classes, 'scores': _scores})
            else:
                predictions.append({'boxes': torch.FloatTensor([[0, 0, 1, 1]]),
                                    'labels': torch.FloatTensor([-1]),
                                    'scores': torch.FloatTensor([0])})

        return predictions
