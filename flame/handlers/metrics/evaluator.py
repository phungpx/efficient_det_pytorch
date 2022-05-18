from typing import Callable
from ignite.metrics import Metric

import torch
from typing import Dict, Tuple


class Evaluator(Metric):
    def __init__(self, eval_fn: Callable, output_transform=lambda x: x):
        super(Evaluator, self).__init__(output_transform)
        self.eval_fn = eval_fn

    def _get_bboxes(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], image_path: str) -> Tuple[list, list]:
        '''
        Args:
            pred: {
                boxes: TensorFloat [N x 4],
                labels: TensorInt64 [N],
                scores: TensorFloat [N],
            }
            target: {
                image_id: TensorInt64 [M],
                boxes: TensorFloat [M x 4],
                labels: TensorInt64 [M],
            }
            image_path: str
        Output:
            detections: List[
                [image_idx, class_prediction, prob_score, [x1, y1, x2, y2], image_path]
            ]

            ground_truths: List[
                [image_idx, class_target, 1, [x1, y1, x2, y2], image_path]
            ]
        '''
        detections, ground_truths = [], []

        image_idx = target['image_id'].item()
        target_boxes = target['boxes'].detach().cpu().numpy().tolist()
        target_labels = target['labels'].detach().cpu().numpy().tolist()

        pred_boxes = pred['boxes'].detach().cpu().numpy().tolist()
        pred_labels = pred['labels'].detach().cpu().numpy().tolist()
        pred_scores = pred['scores'].detach().cpu().numpy().tolist()

        for class_id, bbox in zip(target_labels, target_boxes):
            # [image_idx, class_target, 1, [x1, y1, x2, y2], image_path]
            ground_truth = [image_idx, class_id, 1, bbox, image_path]
            ground_truths.append(ground_truth)

        for class_id, score, bbox in zip(pred_labels, pred_scores, pred_boxes):
            # [image_idx, class_prediction, prob_score, [x1, y1, x2, y2], image_path]
            if class_id == -1 and score == 0:
                continue
            detection = [image_idx, class_id, score, bbox, image_path]
            detections.append(detection)

        return detections, ground_truths

    def reset(self):
        self.detections = []
        self.ground_truths = []

    def update(self, output):
        preds, targets, image_infos = output
        for pred, target, image_info in zip(preds, targets, image_infos):
            _detections, _ground_truths = self._get_bboxes(pred, target, image_info[0])
            self.detections.extend(_detections)
            self.ground_truths.extend(_ground_truths)

    def compute(self):
        metric = self.eval_fn(self.detections, self.ground_truths)
        return metric
