import cv2
import torch
import torchvision
import numpy as np

from pathlib import Path
from ..module import Module
from ignite.engine import Events
from typing import Dict, List


class RegionPredictor(Module):
    def __init__(self,
                 evaluator_name: str = None,
                 compound_coef: int = None,
                 classes: Dict[str, List] = None,
                 thresh_score: float = 0.2,
                 thresh_iou_nms: float = 0.2,
                 output_dir: str = None,
                 output_transform=lambda x: x):
        super(RegionPredictor, self).__init__()
        self.evaluator_name = evaluator_name
        self.compound_coef = compound_coef
        self.classes = classes
        self.thresh_score = thresh_score
        self.thresh_iou_nms = thresh_iou_nms
        self._output_transform = output_transform

        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def init(self):
        assert self.evaluator_name in self.frame, f'The frame does not have {self.evaluator_name}'
        self._attach(self.frame[self.evaluator_name].engine)

    def reset(self):
        pass

    def update(self, output):
        preds, image_infos = output
        image_paths = [image_info[0] for image_info in image_infos]
        image_sizes = [image_info[1] for image_info in image_infos]
        for pred, image_path, image_size in zip(preds, image_paths, image_sizes):
            save_path = str(self.output_dir.joinpath(Path(image_path).name))
            image = cv2.imread(image_path)
            labels, boxes, scores = pred['labels'], pred['boxes'], pred['scores']

            indices = torchvision.ops.nms(boxes, scores, self.thresh_iou_nms)
            labels, boxes, scores = labels[indices], boxes[indices], scores[indices]

            indices = scores > self.thresh_score
            labels, boxes, scores = labels[indices], boxes[indices], scores[indices]

            labels, boxes, scores = labels.data.cpu().numpy(), boxes.data.cpu().numpy(), scores.data.cpu().numpy()

            classes = {label: [cls_name, color] for cls_name, (color, label) in self.classes.items()}
            box_thickness, text_thickness, font_scale = max(image_size) // 200, max(image_size) // 400, max(image_size) / 900
            image_scale = max(image_size) / (512 + self.compound_coef * 128)  # (in this case of preprocessing data using padding to square)
            for (label, box, score) in zip(labels, boxes, scores):
                x1, y1, x2, y2 = np.int32([coord * image_scale for coord in box])
                cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=classes[label][1], thickness=box_thickness)

                title = f"{classes[label][0]}: {score:.4f}"
                ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)

                cv2.rectangle(img=image, pt1=(x1, y1 - int(1.3 * text_height)), pt2=(x1 + text_width, y1), color=(0, 0, 255), thickness=-1)
                cv2.putText(img=image, text=title, org=(x1, y1 - int(0.3 * text_height)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                            color=(255, 255, 255), thickness=text_thickness, lineType=cv2.LINE_AA)

            cv2.imwrite(save_path, image)

    def compute(self):
        pass

    def started(self, engine):
        self.reset()

    @torch.no_grad()
    def iteration_completed(self, engine):
        output = self._output_transform(engine.state.output)
        self.update(output)

    def completed(self, engine):
        self.compute()

    def _attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
