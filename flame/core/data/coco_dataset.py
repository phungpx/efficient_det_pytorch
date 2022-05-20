import cv2
import torch
import random
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from typing import Tuple
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class CoCoDataset(Dataset):
    def __init__(
        self,
        compound_coef: int,
        image_dir: str,
        label_path: str,
        mean: Tuple[float],
        std: Tuple[float],
        transforms: list = None
    ) -> None:
        super(CoCoDataset, self).__init__()
        self.imsize = 512 + compound_coef * 128
        self.transforms = transforms if transforms else []
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)

        self.image_dir = Path(image_dir)
        self.coco = COCO(annotation_file=label_path)
        self.image_indices = self.coco.getImgIds()

        self.class2idx = dict()
        self.coco_label_to_label = dict()
        self.label_to_coco_label = dict()

        categories = self.coco.loadCats(ids=self.coco.getCatIds())
        categories = sorted(categories, key=lambda x: x['id'])
        for category in categories:
            self.label_to_coco_label[len(self.class2idx)] = category['id']
            self.coco_label_to_label[category['id']] = len(self.class2idx)
            self.class2idx[category['name']] = len(self.class2idx)

        self.idx2class = {class_idx: class_name for class_name, class_idx in self.class2idx.items()}

        self.pad_to_square = iaa.PadToSquare(position='right-bottom')

        print(f'{self.image_dir.stem}: {len(self.image_indices)}')
        print(f'All Classes: {self.idx2class}')

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        image, image_info = self.load_image(image_idx=idx)
        boxes, labels = self.load_annot(image_idx=idx)
        if not len(boxes) and not len(labels):
            print(f'Sample {image_info[0]} has no labels')

        bboxes = [BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                  for box, label in zip(boxes, labels)]
        bboxes = BoundingBoxesOnImage(bounding_boxes=bboxes, shape=image.shape)
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bboxes = transform(image=image, bounding_boxes=bboxes)

        # Rescale image and bounding boxes
        image, bboxes = self.pad_to_square(image=image, bounding_boxes=bboxes)
        image, bboxes = iaa.Resize(size=self.imsize)(image=image, bounding_boxes=bboxes)

        bboxes = bboxes.on(image)

        boxes = [[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bboxes.bounding_boxes]
        labels = [bbox.label for bbox in bboxes.bounding_boxes]

        # Convert to Torch Tensor
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)  # suppose all instances are not crowd
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Target
        target = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
        }

        # Sample
        sample = torch.from_numpy(np.ascontiguousarray(image))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = (sample.float().div(255.) - self.mean) / self.std

        return sample, target, image_info

    def load_image(self, image_idx):
        image_info = self.coco.loadImgs(ids=self.image_indices[image_idx])[0]
        image_path = str(self.image_dir.joinpath(image_info['file_name']))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_info = [image_path, image.shape[1::-1]]

        return image, image_info

    def load_annot(self, image_idx):
        boxes, labels = [], []
        annot_indices = self.coco.getAnnIds(imgIds=self.image_indices[image_idx], iscrowd=False)
        if not len(annot_indices):
            labels, boxes = [-1], [[0, 0, 1, 1]]
            return boxes, labels

        annot_infos = self.coco.loadAnns(ids=annot_indices)
        for idx, annot_info in enumerate(annot_infos):
            # some annotations have basically no width or height, skip them.
            if annot_info['bbox'][2] < 1 or annot_info['bbox'][3] < 1:
                continue

            bbox = self.xywh2xyxy(annot_info['bbox'])
            label = self.coco_label_to_label[annot_info['category_id']]
            boxes.append(bbox)
            labels.append(label)

        return boxes, labels

    def xywh2xyxy(self, box):
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        return box

    def image_aspect_ratio(self, image_idx):
        image_info = self.coco.loadImgs(self.image_indices[image_idx])[0]
        return float(image_info['width']) / float(image_info['height'])

    @property
    def num_classes(self):
        return len(list(self.idx2class.keys()))
