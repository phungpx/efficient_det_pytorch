import torch
import torch.nn as nn
from collections import Counter
from ignite.metrics import Metric


class mAP(Metric):
    def __init__(self, num_classes, iou_threshold, box_format='corners', output_transform=lambda x: x):
        super(mAP, self).__init__(output_transform)
        self.map = MeanAveragePrecision(num_classes=num_classes, iou_threshold=iou_threshold, box_format=box_format)

    def _get_box_info(self, pred, target):
        detections, groundtruths = [], []

        image_idx = target['image_id'].item()
        target_boxes = target['boxes'].detach().cpu().numpy()
        target_labels = target['labels'].detach().cpu().numpy()

        pred_boxes = pred['boxes'].detach().cpu().numpy()
        pred_labels = pred['labels'].detach().cpu().numpy()
        pred_scores = pred['scores'].detach().cpu().numpy()

        for class_id, bbox in zip(target_labels, target_boxes):
            # [image_idx, class_target, 1, x1, y1, x2, y2]
            groundtruth = [image_idx, class_id, 1, bbox[0], bbox[1], bbox[2], bbox[3]]
            groundtruths.append(groundtruth)

        for class_id, score, bbox in zip(pred_labels, pred_scores, pred_boxes):
            # [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
            if class_id == -1 and score == 0:
                continue
            detection = [image_idx, class_id, score, bbox[0], bbox[1], bbox[2], bbox[3]]
            detections.append(detection)

        return detections, groundtruths

    def reset(self):
        self.detections = []
        self.groundtruths = []

    def update(self, output):
        preds, targets = output
        for pred, target in zip(preds, targets):
            _detections, _groundtruths = self._get_box_info(pred, target)
            if len(_detections) and len(_groundtruths):
                self.detections.extend(_detections)
                self.groundtruths.extend(_groundtruths)

    def compute(self):
        metric = self.map(self.detections, self.groundtruths)
        return metric


class MeanAveragePrecision(nn.Module):
    def __init__(self, num_classes, iou_threshold=0.5, box_format='corners'):
        super(MeanAveragePrecision, self).__init__()
        self.epsilon = 1e-6  # used for numerical stability later on
        self.box_format = box_format
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

    def _compute_iou(self, boxes_preds, boxes_labels, box_format="midpoint"):
        """Calculates intersection over union
        Parameters:
            boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        Returns:
            tensor: Intersection over union for all examples
        """
        if box_format == "midpoint":
            box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
            box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
            box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
            box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

            box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
            box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
            box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
            box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

        elif box_format == "corners":
            box1_x1 = boxes_preds[..., 0:1]
            box1_y1 = boxes_preds[..., 1:2]
            box1_x2 = boxes_preds[..., 2:3]
            box1_y2 = boxes_preds[..., 3:4]

            box2_x1 = boxes_labels[..., 0:1]
            box2_y1 = boxes_labels[..., 1:2]
            box2_x2 = boxes_labels[..., 2:3]
            box2_y2 = boxes_labels[..., 3:4]

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        # Need clamp(0) in case they do not intersect, then we want intersection to be 0
        intersection = (x2 - x1).clamp(min=0.) * (y2 - y1).clamp(min=0.)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + self.epsilon)

    def mean_average_precision(self, pred_boxes, target_boxes,
                               iou_threshold=0.5,
                               box_format="midpoint",
                               num_classes=20):
        """
        Calculates mean average precision.
        Parameters:
            pred_boxes (list): list of lists containing all bboxes with each bboxes.
            specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2].
            target_boxes (list): Similar as pred_boxes except all the correct ones.
            iou_threshold (float): threshold where predicted bboxes is correct.
            box_format (str): "midpoint" or "corners" used to specify bboxes.
            num_classes (int): number of classes.
        Returns:
            float: mAP value across all classes given a specific IoU threshold.
        """
        mAP = 0.
        average_precisions = []  # list storing all AP for respective classes
        for c in range(num_classes):
            # go through all predictions and targets.
            # and only add the ones that belong to the current class c.
            detections = [detection for detection in pred_boxes if detection[1] == c]
            groundtruths = [groundtruth for groundtruth in target_boxes if groundtruth[1] == c]

            # find the amount of bboxes for each training example.
            # Counter here finds how many ground truth bboxes we get for each training example, so let's say img 0 has 3.
            # img 1 has 5 then we will obtain a dictionary with: amount_bboxes = {0:3, 1:5}.
            amount_bboxes = Counter([groundtruth[0] for groundtruth in groundtruths])

            # we then go through each key, val in this dictionary and convert to the following (w.r.t same example):
            # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for image_idx, num_boxes in amount_bboxes.items():
                amount_bboxes[image_idx] = torch.zeros(num_boxes)

            # sort by box probabilities which is index 2
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_target_bboxes = len(groundtruths)

            # if none exists for this class then we can safely skip
            if total_target_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                # only take out the groundtruths that have the same
                # training idx as detection

                _image_groundtruths = [groundtruth for groundtruth in groundtruths if groundtruth[0] == detection[0]]

                best_iou = 0
                for idx, groundtruth in enumerate(_image_groundtruths):
                    iou = self._compute_iou(torch.tensor(detection[3:]),
                                            torch.tensor(groundtruth[3:]),
                                            box_format=box_format)

                    if iou > best_iou:
                        best_iou = iou
                        best_groundtruth_idx = idx

                if best_iou > self.iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[detection[0]][best_groundtruth_idx] == 0:
                        # true positive and add this bounding box to seen
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_groundtruth_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_target_bboxes + self.epsilon)
            recalls = torch.cat((torch.tensor([0]), recalls))
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + self.epsilon)
            precisions = torch.cat((torch.tensor([1]), precisions))
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))

        mAP = sum(average_precisions) / len(average_precisions) if len(average_precisions) else 0.

        return mAP

    def forward(self, pred_boxes, target_boxes):
        mAP = self.mean_average_precision(pred_boxes, target_boxes,
                                          iou_threshold=self.iou_threshold,
                                          box_format=self.box_format,
                                          num_classes=self.num_classes)
        return mAP
