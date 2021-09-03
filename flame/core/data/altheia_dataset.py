import cv2
import torch
import random
import numpy as np
import imgaug.augmenters as iaa
import xml.etree.ElementTree as ET

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class AltheiaDataset(Dataset):
    def __init__(self, dirname: str = None,
                 image_patterns: List[str] = ['*.jpg'],
                 label_patterns: List[str] = ['*.xml'],
                 classes: Dict[str, int] = None,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 compound_coef: int = 0,
                 transforms: Optional[List] = None):
        super(AltheiaDataset, self).__init__()
        self.classes = classes
        self.imsize = 512 + compound_coef * 128
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)

        self.transforms = transforms if transforms else []

        image_paths, label_paths = [], []
        for image_pattern in image_patterns:
            image_paths.extend(Path(dirname).glob('**/{}'.format(image_pattern)))
        for label_pattern in label_patterns:
            label_paths.extend(Path(dirname).glob('**/{}'.format(label_pattern)))

        image_paths = natsorted(image_paths, key=lambda x: str(x.stem))
        label_paths = natsorted(label_paths, key=lambda x: str(x.stem))

        self.data_pairs = [[image, label] for image, label in zip(image_paths, label_paths)]

        self.pad_to_square = iaa.PadToSquare(position='right-bottom')

        print(f'{Path(dirname).stem}: {len(self.data_pairs)}')

    def __len__(self):
        return len(self.data_pairs)

    def _get_label_info(self, lable_path: str, classes: dict) -> Dict:
        root = ET.parse(str(lable_path)).getroot()
        page = root.find('{}Page'.format(''.join(root.tag.partition('}')[:2])))
        width, height = int(page.get('imageWidth')), int(page.get('imageHeight'))

        label_info = []
        for card_type, label in classes.items():
            regions = root.findall('.//*[@value=\"{}\"]/../..'.format(card_type)) + root.findall('.//*[@name=\"{}\"]/../..'.format(card_type))
            for region in regions:
                points = [[int(float(coord)) for coord in point.split(',')] for point in region[0].get('points').split()]
                # assert len(points) >= 4, 'Length of points must be greater than or equal 4.'
                mask = np.zeros(shape=(height, width), dtype=np.uint8)
                cv2.fillPoly(img=mask, pts=np.int32([points]), color=(255, 255, 255))

                x1 = min([point[0] for point in points])
                y1 = min([point[1] for point in points])
                x2 = max([point[0] for point in points])
                y2 = max([point[1] for point in points])
                bbox = (x1, y1, x2, y2)

                label_info.append({'mask': mask, 'label': label, 'bbox': bbox})

        if not len(label_info):
            label_info.append({'mask': np.zeros(shape=(height, width), dtype=np.uint8),
                               'label': -1,
                               'bbox': (0, 0, 1, 1)})

        return label_info

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, Tuple[str, Tuple[int, int]]]:
        image_path, label_path = self.data_pairs[idx]
        label_info = self._get_label_info(lable_path=str(label_path), classes=self.classes)

        image = cv2.imread(str(image_path))
        image_info = (str(image_path), image.shape[1::-1])  # image path, (w, h)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = [label['bbox'] for label in label_info]
        labels = [label['label'] for label in label_info]

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
