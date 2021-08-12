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


class VOC2007Dataset(Dataset):
    def __init__(self, image_dir: str = None, label_dir: str = None,
                 image_pattern: str = '*.jpg', label_pattern: str = '*.xml',
                 classes: Dict[str, int] = None,
                 mean: List[float] = [0.485, 0.456, 0.406], std: List[float] = [0.229, 0.224, 0.225],
                 compound_coef: int = 0, transforms: Optional[List] = None):
        super(VOC2007Dataset, self).__init__()
        self.classes = classes
        self.imsize = 512 + compound_coef * 128
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)

        self.transforms = transforms if transforms else []

        image_paths = natsorted(list(Path(image_dir).glob(f'{image_pattern}')), key=lambda x: str(x.stem))
        label_paths = natsorted(list(Path(label_dir).glob(f'{label_pattern}')), key=lambda x: str(x.stem))

        self.data_pairs = [[image, label] for image, label in zip(image_paths, label_paths)]

        self.pad_to_square = iaa.PadToSquare(position='right-bottom')

        print(f'{Path(image_dir).stem}: {len(self.data_pairs)}')

    def __len__(self):
        return len(self.data_pairs)

    def _get_label_info(self, label_path):
        tree = ET.parse(str(label_path))
        image_info = {'image_name': tree.find('filename').text,
                      'height': int(tree.find('size').find('height').text),
                      'width': int(tree.find('size').find('width').text),
                      'depth': int(tree.find('size').find('depth').text)}
        label_info = []
        objects = tree.findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            bbox = np.int32([bndbox.find('xmin').text, bndbox.find('ymin').text,
                             bndbox.find('xmax').text, bndbox.find('ymax').text])
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

        # Pad to square to keep object's ratio
        bbs = BoundingBoxesOnImage([BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                                    for box, label in zip(boxes, labels)], shape=image.shape)
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bbs = transform(image=image, bounding_boxes=bbs)

        # Rescale image and bounding boxes
        image, bbs = self.pad_to_square(image=image, bounding_boxes=bbs)
        sample, bbs = iaa.Resize(size=self.imsize)(image=image, bounding_boxes=bbs)
        bbs = bbs.on(sample)

        # Convert from Bouding Box Object to boxes, labels list
        boxes = [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]
        labels = [bb.label for bb in bbs.bounding_boxes]

        # Convert to Torch Tensor
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx], dtype=torch.int64)

        # # Target
        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id}

        # Image
        sample = torch.from_numpy(np.ascontiguousarray(sample))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = (sample.float().div(255.) - self.mean) / self.std

        return sample, target, image_info
