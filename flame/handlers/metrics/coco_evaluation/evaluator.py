import torch
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from typing import List
from ignite.metrics import Metric

from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset


class Evaluator(Metric):
    def __init__(
        self,
        iou_types: List[str] = ['bbox', 'segm', 'keypoints'],
        output_transform=lambda x: (x[0], x[1], x[3])
    ):
        super(Evaluator, self).__init__(output_transform)
        coco = get_coco_api_from_dataset(dataloader.dataset)
        self.coco_evaluator = CocoEvaluator(coco, iou_types)

    def reset(self):
        self.detections = []
        self.ground_truths = []

    def update(self, output):
        preds, targets, image_infos = output
        preds = [{k: v.to(torch.device('cpu')) for k, v in pred.items()} for pred in preds]
        targets = [{k: v.to(torch.device('cpu')) for k, v in target.items()} for target in targets]
        results = {target['image_id'].item(): pred for target, pred in zip(targets, preds)}
        self.coco_evaluator.update(results)

    def compute(self):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        return self.coco_evaluator

    def convert_to_coco_api(self, dataloader):
        coco_ds = COCO()
        # annotation IDs need to start at 1, not 0, see torchvision issue #1530
        ann_id = 1
        dataset = {'images': [], 'categories': [], 'annotations': []}
        categories = set()
        for img_idx in range(len(ds)):
            # find better way to get target
            # targets = ds.get_annotations(img_idx)
            img, targets = ds[img_idx]
            image_id = targets["image_id"].item()
            img_dict = {}
            img_dict['id'] = image_id
            img_dict['height'] = img.shape[-2]
            img_dict['width'] = img.shape[-1]
            dataset['images'].append(img_dict)
            bboxes = targets["boxes"]
            bboxes[:, 2:] -= bboxes[:, :2]
            bboxes = bboxes.tolist()
            labels = targets['labels'].tolist()
            areas = targets['area'].tolist()
            iscrowd = targets['iscrowd'].tolist()
            if 'masks' in targets:
                masks = targets['masks']
                # make masks Fortran contiguous for coco_mask
                masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
            if 'keypoints' in targets:
                keypoints = targets['keypoints']
                keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
            num_objs = len(bboxes)
            for i in range(num_objs):
                ann = {}
                ann['image_id'] = image_id
                ann['bbox'] = bboxes[i]
                ann['category_id'] = labels[i]
                categories.add(labels[i])
                ann['area'] = areas[i]
                ann['iscrowd'] = iscrowd[i]
                ann['id'] = ann_id
                if 'masks' in targets:
                    ann["segmentation"] = coco_mask.encode(masks[i].numpy())
                if 'keypoints' in targets:
                    ann['keypoints'] = keypoints[i]
                    ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
                dataset['annotations'].append(ann)
                ann_id += 1
        dataset['categories'] = [{'id': i} for i in sorted(categories)]
        coco_ds.dataset = dataset
        coco_ds.createIndex()
        return coco_ds
