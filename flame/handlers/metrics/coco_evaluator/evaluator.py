import json
import numpy as np
from pathlib import Path
from ignite.metrics import Metric
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Evaluator(Metric):
    def __init__(self, compound_coef: int, annotation_file: str, save_path: str, output_transform=lambda x: x):
        super(Evaluator, self).__init__(output_transform)
        self.save_path = Path(save_path)
        self.imsize = compound_coef * 128 + 512
        self.groundtruth_coco = COCO(annotation_file=annotation_file)

    def reset(self):
        self.predictions = []

    def update(self, output):
        preds, targets, image_infos = output
        for pred, target, image_info in zip(preds, targets, image_infos):
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()

            scale = max(image_info[1]) / self.imsize

            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
            boxes = boxes.astype(np.int32)

            for box, score, label in zip(boxes, scores, labels):
                if label != -1:
                    prediction = {
                        'image_id': target['image_id'].item(),
                        'category_id': int(label.item()) + 1,
                        'score': float(score.item()),
                        'bbox': box.tolist(),
                    }

                    self.predictions.append(prediction)

    def compute(self):
        if not len(self.predictions):
            raise Exception('the model does not provide any valid output,\
                            check model architecture and the data input')

        if self.save_path.exists():
            self.save_path.unlink()
        with open(file=str(self.save_path), mode='w', encoding='utf-8') as f:
            json.dump(self.predictions, f, ensure_ascii=False, indent=4)

        detection_coco = self.groundtruth_coco.loadRes(str(self.save_path))

        coco_eval = COCOeval(self.groundtruth_coco, detection_coco, 'bbox')
        coco_eval.params.imgIds = self.groundtruth_coco.getImgIds()
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
