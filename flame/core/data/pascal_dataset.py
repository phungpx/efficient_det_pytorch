import cv2
import torch
import random
import numpy as np
import imgaug.augmenters as iaa
import xml.etree.ElementTree as ET

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class PascalDataset(Dataset):
    def __init__(
        self,
        VOC2007: Dict[str, str] = None,
        VOC2012: Dict[str, str] = None,
        image_extent: str = '.jpg',
        label_extent: str = '.xml',
        classes: Dict[str, int] = None,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        compound_coef: int = 0,
        transforms: Optional[List] = None
    ):
        super(PascalDataset, self).__init__()
        self.classes = classes
        self.imsize = 512 + compound_coef * 128
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.pad_to_square = iaa.PadToSquare(position='right-bottom')
        self.transforms = transforms if transforms else []

        # VOC2007
        image_dir, label_dir = Path(VOC2007['image_dir']), Path(VOC2007['label_dir'])
        image_paths = natsorted(list(Path(image_dir).glob(f'*{image_extent}')), key=lambda x: str(x.stem))
        label_paths = natsorted(list(Path(label_dir).glob(f'*{label_extent}')), key=lambda x: str(x.stem))
        voc2007_pairs = [[image, label] for image, label in zip(image_paths, label_paths) if image.stem == label.stem]

        # VOC2012
        image_dir, label_dir, txt_path = Path(VOC2012['image_dir']), Path(VOC2012['label_dir']), Path(VOC2012['txt_path'])
        with txt_path.open(mode='r', encoding='utf-8') as fp:
            image_names = fp.read().splitlines()

        voc2012_pairs = []
        for image_name in image_names:
            image_path = image_dir.joinpath(f'{image_name}{image_extent}')
            label_path = label_dir.joinpath(f'{image_name}{label_extent}')
            if image_path.exists() and label_path.exists():
                voc2012_pairs.append([image_path, label_path])

        self.data_pairs = voc2007_pairs + voc2012_pairs

        print(f'- {txt_path.stem}:')
        print(f'\t VOC2007: {len(voc2007_pairs)}')
        print(f'\t VOC2012: {len(voc2012_pairs)}')
        print(f'\t Total: {len(self.data_pairs)}')

    def __len__(self):
        return len(self.data_pairs)

    def _get_label_info(self, label_path):
        tree = ET.parse(str(label_path))
        image_info = {
            'image_name': tree.find('filename').text,
            'height': int(tree.find('size').find('height').text),
            'width': int(tree.find('size').find('width').text),
            'depth': int(tree.find('size').find('depth').text)
        }

        label_info = []
        objects = tree.findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            bbox = np.int32(
                [
                    bndbox.find('xmin').text,
                    bndbox.find('ymin').text,
                    bndbox.find('xmax').text,
                    bndbox.find('ymax').text,
                ]
            )
            label_name = obj.find('name').text
            label_info.append({'label': label_name, 'bbox': bbox})

        return image_info, label_info

    def __getitem__(self, idx):
        image_path, label_path = self.data_pairs[idx]
        _, label_info = self._get_label_info(label_path)

        image = cv2.imread(str(image_path))
        image_info = [str(image_path), image.shape[1::-1]]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = [label['bbox'] for label in label_info]
        labels = [self.classes[label['label']] for label in label_info]

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
