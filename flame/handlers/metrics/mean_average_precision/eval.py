from typing import Callable
from ignite.metrics import Metric


class Evaluator(Metric):
    def __init__(self, eval_fn: Callable, output_transform=lambda x: x):
        super(Evaluator, self).__init__(output_transform)
        self.eval_fn = eval_fn

    def _get_bboxes(self, pred, target):
        detections, ground_truths = [], []

        image_idx = target['image_id'].item()
        target_boxes = target['boxes'].detach().cpu().numpy().tolist()
        target_labels = target['labels'].detach().cpu().numpy().tolist()

        pred_boxes = pred['boxes'].detach().cpu().numpy().tolist()
        pred_labels = pred['labels'].detach().cpu().numpy().tolist()
        pred_scores = pred['scores'].detach().cpu().numpy().tolist()

        for class_id, bbox in zip(target_labels, target_boxes):
            # [image_idx, class_target, 1, [x1, y1, x2, y2]]
            ground_truth = [image_idx, class_id, 1, bbox]
            ground_truths.append(ground_truth)

        for class_id, score, bbox in zip(pred_labels, pred_scores, pred_boxes):
            # [train_idx, class_prediction, prob_score, [x1, y1, x2, y2]]
            if class_id == -1 and score == 0:
                continue
            detection = [image_idx, class_id, score, bbox]
            detections.append(detection)

        return detections, ground_truths

    def reset(self):
        self.detections = []
        self.ground_truths = []

    def update(self, output):
        preds, targets = output
        for pred, target in zip(preds, targets):
            _detections, _ground_truths = self._get_bboxes(pred, target)
            self.detections.extend(_detections)
            self.ground_truths.extend(_ground_truths)

    def compute(self):
        metric = self.eval_fn(self.detections, self.ground_truths)
        return metric
