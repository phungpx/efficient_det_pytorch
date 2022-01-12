import cv2
import json
import torch
import random
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class LabelmeDataset(Dataset):
    def __init__(
        self,
        dirnames: List[str] = None,
        image_patterns: List[str] = ['*.jpg'],
        label_patterns: List[str] = ['*.json'],
        classes: Dict[str, int] = None,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        compound_coef: int = 0,
        transforms: Optional[List] = None
    ) -> None:
        super(LabelmeDataset, self).__init__()
        self.classes = classes
        self.imsize = 512 + compound_coef * 128
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.pad_to_square = iaa.PadToSquare(position='right-bottom')

        self.transforms = transforms if transforms else []

        image_paths, label_paths = [], []
        for dirname in dirnames:
            for image_pattern in image_patterns:
                image_paths.extend(Path(dirname).glob(f'**/{image_pattern}'))
            for label_pattern in label_patterns:
                label_paths.extend(Path(dirname).glob(f'**/{label_pattern}'))

        image_paths = natsorted(image_paths, key=lambda x: str(x.stem))
        label_paths = natsorted(label_paths, key=lambda x: str(x.stem))

        self.data_pairs = [[image, label] for image, label in zip(image_paths, label_paths)]

        print(f'{Path(dirnames[0]).parent.stem}: {len(self.data_pairs)}')

    def __len__(self):
        return len(self.data_pairs)

    def _get_label_info(self, lable_path: str, classes: dict) -> Dict:
        with open(file=lable_path, mode='r', encoding='utf-8') as f:
            json_info = json.load(f)

        label_info = []
        for shape in json_info['shapes']:
            label = shape['label']
            points = shape['points']
            if label in self.classes and len(points) > 0:
                x1 = min([point[0] for point in points])
                y1 = min([point[1] for point in points])
                x2 = max([point[0] for point in points])
                y2 = max([point[1] for point in points])
                bbox = (x1, y1, x2, y2)

                label_info.append({'label': self.classes[label], 'bbox': bbox})

        if not len(label_info):
            label_info.append({'label': -1, 'bbox': (0, 0, 1, 1)})

        return label_info

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, Tuple[str, Tuple[int, int]]]:
        image_path, label_path = self.data_pairs[idx]
        label_info = self._get_label_info(lable_path=str(label_path), classes=self.classes)

        image = cv2.imread(str(image_path))
        image_info = (str(image_path), image.shape[1::-1])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = [label['bbox'] for label in label_info]
        labels = [label['label'] for label in label_info]

        bbs = BoundingBoxesOnImage(
            [
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                for box, label in zip(boxes, labels)
            ],
            shape=image.shape
        )

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bbs = transform(image=image, bounding_boxes=bbs)

        # Pad to square to keep object's ratio, then Rescale image and bounding boxes
        image, bbs = self.pad_to_square(image=image, bounding_boxes=bbs)
        sample, bbs = iaa.Resize(size=self.imsize)(image=image, bounding_boxes=bbs)
        bbs = bbs.on(sample)

        # Convert from Bouding Box Object to boxes, labels list
        boxes = [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]
        labels = [bb.label for bb in bbs.bounding_boxes]

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

        # Image
        sample = torch.from_numpy(np.ascontiguousarray(sample))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = (sample.float().div(255.) - self.mean) / self.std

        return sample, target, image_info
