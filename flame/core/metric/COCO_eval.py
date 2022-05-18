import json
import numpy as np
from pathlib import Path
from ignite.metrics import Metric
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from typing import List, Dict, Optional, Callable, Union


class COCOEvaluator(Metric):
    def __init__(
        self,
        compound_coef: Optional[int] = None,
        classes: Optional[Union[Dict[str, int], int]] = None,
        annotation_file: Optional[str] = None,
        label_to_coco_label: Optional[Dict[int, int]] = {None: 0},
        annotation_type: str = 'bbox',
        detection_path: str = None,
        ground_truth_path: str = None,
        output_transform: Callable = lambda x: x
    ):
        super(COCOEvaluator, self).__init__(output_transform)
        self.detection_path = detection_path
        self.ground_truth_path = ground_truth_path

        self.imsize = 512 + compound_coef * 128 if compound_coef is not None else None

        if isinstance(classes, int):
            classes = {i: i for i in range(classes)}
        self.classes = classes

        self.annotation_file = annotation_file
        self.label_to_coco_label = label_to_coco_label

        if annotation_type in ['segm','bbox','keypoints']:
            self.annotation_type = annotation_type
        else:
            print('Annotation Type is invalid.')

    def reset(self):
        self.detections: List[Dict] = []  # List[{'image_id': ..., 'category_id': ..., 'score': ..., 'bbox': ...}]

        if self.annotation_file is None:
            self.annot_id = 0
            # initialize groundtruth in COCO format
            self.ground_truths: Dict = {
                'annotations': [],
                'images': [],
                'categories': [
                    {'id': class_id, 'name': class_name}
                    for class_name, class_id in self.classes.items()
                ]
            }

    def update(self, output):
        preds, targets, image_infos = output
        for pred, target, image_info in zip(preds, targets, image_infos):
            pred_boxes = pred['boxes'].cpu().numpy()  # format x1, y1, x2, y2
            pred_labels = pred['labels'].cpu().numpy().tolist()
            pred_scores = pred['scores'].cpu().numpy().tolist()

            image_id = target['image_id'].item()
            scale = max(image_info[1]) / self.imsize if self.imsize is not None else 1.  # deal with input sample is paded to square (bottom-right)

            pred_boxes[:, [2, 3]] -= pred_boxes[:, [0, 1]]  # convert x1, y1, x2, y2 to x1, y1, w, h
            pred_boxes = (pred_boxes * scale).astype(np.int32).tolist()  # scale boxes to orginal size.

            # detection
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if label == -1:
                    continue

                self.detections.append(
                    {
                        'image_id': image_id,
                        'category_id': self.label_to_coco_label.get(label, label),
                        'score': score,
                        'bbox': box
                    }
                )

            if self.annotation_file is None:
                true_labels = target['labels'].cpu().numpy().tolist()
                true_areas = target['area'].cpu().numpy().tolist()

                true_boxes = target['boxes'].cpu().numpy()  # format x1, y1, x2, y2
                true_boxes[:, [2, 3]] -= true_boxes[:, [0, 1]]  # convert x1, y1, x2, y2 to x1, y1, w, h
                true_boxes = (true_boxes * scale).astype(np.int32).tolist()  # scale boxes to orginal size.

                # create ground truth in COCO format if has no COCO ground truth file.
                for box, label, area in zip(true_boxes, true_labels, true_areas):
                    annotation = {
                        'image_id': image_id,
                        'category_id': label,
                        'bbox': box,
                        'iscrowd': 0,
                        'area': area,
                        'id': self.annot_id,
                    }
                    self.ground_truths['annotations'].append(annotation)
                    self.annot_id += 1

                self.ground_truths['images'].append(
                    {
                        'file_name': image_info[0],
                        'height': image_info[1][1],
                        'width': image_info[1][0],
                        'id': image_id,
                    }
                )

    def compute(self):
        if not len(self.detections):
            raise Exception('the model does not provide any valid output,\
                            check model architecture and the data input')

        # Create Ground Truth COCO Format
        if self.annotation_file is not None:
            groundtruth_coco = COCO(annotation_file=self.annotation_file)
        else:
            with open(file=self.ground_truth_path, mode='w', encoding='utf-8') as f:
                json.dump(self.ground_truths, f, ensure_ascii=False, indent=4)
            # save ground truth to json file and then load to COCO class
            groundtruth_coco = COCO(annotation_file=self.ground_truth_path)

        # Create Detection COCO Format
        if Path(self.detection_path).exists():
            Path(self.detection_path).unlink()

        with open(file=self.detection_path, mode='w', encoding='utf-8') as f:
            json.dump(self.detections, f, ensure_ascii=False, indent=4)

        # using COCO object to load detections.
        detection_coco = groundtruth_coco.loadRes(self.detection_path)

        # Evaluation
        coco_eval = COCOeval(groundtruth_coco, detection_coco, self.annotation_type)
        coco_eval.params.imgIds = groundtruth_coco.getImgIds()
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
