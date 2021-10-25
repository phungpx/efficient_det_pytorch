from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import torch
import numpy as np
from torch import nn

import utils


def chunks(lst: list, size: Optional[int] = None) -> Union[List, Generator]:
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i:i + size]


class Detector:
    def __init__(
        self,
        model: nn.Module,
        compound_coef: int,
        weight_path: str,
        classes: Dict[int, str],
        mean: List[float], std: List[float],
        batch_size: Optional[int] = 1,
        device: str = 'cpu'
    ) -> None:
        super(Detector, self).__init__()
        self.model = model
        self.device = device
        self.classes = classes
        self.batch_size = batch_size
        self.imsize = 512 + 128 * compound_coef

        self.mean = torch.tensor(mean, dtype=torch.float, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float, device=device).view(1, 3, 1, 1)

        state_dict = torch.load(f=utils.abs_path(weight_path), map_location='cpu')
        self.model.load_state_dict(state_dict=state_dict)
        self.model.eval().to(device)

    def __call__(self, images: List[np.ndarray]) -> List[Dict]:
        original_sizes, samples = self.preprocess(images)
        original_sizes, preds = self.process(original_sizes, samples)
        outputs = self.postprocess(original_sizes, preds)
        return outputs

    def preprocess(
        self, images: List[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        original_sizes, samples = [], []
        for image in images:
            original_sizes.append((image.shape[1], image.shape[0]))
            sample = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sample = self.resize(sample)
            sample = self.pad_to_square(sample)
            samples.append(sample)

        return original_sizes, samples

    def process(
        self, original_sizes: List[Tuple[int, int]], samples: List[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        preds = []
        for batch in chunks(samples, size=self.batch_size):
            batch = [torch.from_numpy(image) for image in batch]
            batch = torch.stack(batch, dim=0).to(self.device)
            batch = batch.permute(0, 3, 1, 2).contiguous()
            batch = (batch.float().div(255.) - self.mean) / self.std

            with torch.no_grad():
                preds += self.model.inference(batch)

        return original_sizes, preds

    def postprocess(
        self, original_sizes: List[Tuple[int, int]], preds: List[Dict]
    ) -> List[Dict]:

        for original_size, pred in zip(original_sizes, preds):
            pred['boxes'] *= max(original_size) / self.imsize
            pred['names'] = [self.classes.get(label.item(), 'background') for label in pred['labels']]

        return preds

    def pad_to_square(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        max_size = max(height, width)
        image = np.pad(image, ((0, max_size - height), (0, max_size - width), (0, 0)))
        return image

    def resize(self, image: np.ndarray) -> np.ndarray:
        ratio = self.imsize / max(image.shape)
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
        return image
