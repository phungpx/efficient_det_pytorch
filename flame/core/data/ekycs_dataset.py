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
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class EkycDataset(Dataset):
    def __init__(self, dirname: str = None,
                 image_patterns: List[str] = ['*.jpg'],
                 label_patterns: List[str] = ['*.xml'],
                 classes: Dict[str, int] = None,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 compound_coef: int = 0,
                 transforms: Optional[List] = None):
        super(EkycDataset, self).__init__()
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

    def _label_info(self, lable_path: str, classes: dict) -> Tuple[List[np.ndarray], List[int]]:
        root = ET.parse(str(lable_path)).getroot()
        page = root.find('{}Page'.format(''.join(root.tag.partition('}')[:2])))
        width, height = int(page.get('imageWidth')), int(page.get('imageHeight'))

        masks, labels = [], []
        for card_type, label in classes.items():
            regions = root.findall('.//*[@value=\"{}\"]/../..'.format(card_type)) + root.findall('.//*[@name=\"{}\"]/../..'.format(card_type))
            for region in regions:
                points = [[int(float(coord)) for coord in point.split(',')] for point in region[0].get('points').split()]
                # assert len(points) >= 4, 'Length of points must be greater than or equal 4.'
                mask = np.zeros(shape=(height, width), dtype=np.uint8)
                cv2.fillPoly(img=mask, pts=np.int32([points]), color=(255, 255, 255))
                masks.append(mask)
                labels.append(label)

        if (not len(masks)) and (not len(labels)):
            masks, labels = [np.zeros(shape=(height, width), dtype=np.uint8)], [-1]

        return masks, labels

    def __getitem__(self, idx):
        image_path, label_path = self.data_pairs[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks, labels = self._label_info(lable_path=str(label_path), classes=self.classes)
        if (not len(masks)) and (not len(labels)):
            raise ValueError('image {} has no label.'.format(image_path.stem))

        image_info = [str(image_path), image.shape[1::-1]]

        # create SegmentationMapsOnImage
        masks = [SegmentationMapsOnImage(mask, image.shape[:2]) for mask in masks]

        # transform masks and image
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            _transform = transform.to_deterministic()
            image = _transform(image=image)
            masks = [_transform(segmentation_maps=mask) for mask in masks]
        masks = [mask.get_arr() for mask in masks]

        # padding image, masks to square and then resize image and masks
        image = cv2.resize(self.pad_to_square(image=image), dsize=(self.imsize, self.imsize))
        masks = [cv2.resize(self.pad_to_square(image=mask), dsize=(self.imsize, self.imsize)) for mask in masks]

        # get boxes in masks
        boxes = []
        for i, mask in enumerate(masks):
            pos = np.where(mask == 255)
            if len(pos) != 0 and len(pos[1]) != 0:
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                boxes.append([0, 0, 1, 1])
                labels[i] = -1

        # Convert to Torch Tensor
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx], dtype=torch.int64)

        # Target
        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id}

        # Image
        sample = torch.from_numpy(np.ascontiguousarray(image))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = (sample.float().div(255.) - self.mean) / self.std

        return sample, target, image_info
