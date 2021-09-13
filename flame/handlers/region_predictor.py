import cv2
import torch
import torchvision
import numpy as np

from pathlib import Path
from ..module import Module
from ignite.engine import Events
from typing import Dict, List, Optional


class RegionPredictor(Module):
    def __init__(self,
                 evaluator_name: str = None,
                 compound_coef: int = None,
                 classes: Dict[str, List] = None,
                 score_threshold: Optional[float] = None,
                 iou_threshold: Optional[float] = None,
                 output_dir: str = None,
                 output_transform=lambda x: x):
        super(RegionPredictor, self).__init__()
        self.classes = classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.evaluator_name = evaluator_name
        self.imsize = 512 + compound_coef * 128
        self.output_transform = output_transform
        self.output_dir = Path(output_dir)

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
            save_dir = self.output_dir.joinpath(Path(image_path).parent.stem)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            save_path = str(save_dir.joinpath(Path(image_path).name))

            image = cv2.imread(image_path)

            labels, boxes, scores = pred['labels'], pred['boxes'], pred['scores']

            if self.score_threshold:
                indices = scores > self.score_threshold
                labels, boxes, scores = labels[indices], boxes[indices], scores[indices]

            if self.iou_threshold:
                indices = torchvision.ops.nms(boxes, scores, self.iou_threshold)
                labels, boxes, scores = labels[indices], boxes[indices], scores[indices]

            boxes = boxes.data.cpu().numpy().tolist()
            labels = labels.data.cpu().numpy().tolist()
            scores = scores.data.cpu().numpy().tolist()

            if self.classes:
                classes = {
                    label: [cls_name, color] for cls_name, (color, label) in self.classes.items()
                }

            font_scale = max(image_size) / 1200
            box_thickness = max(image_size) // 400
            text_thickness = max(image_size) // 600
            image_scale = max(image_size) / self.imsize  # in this case of preprocessing data using padding to square.

            for (label, box, score) in zip(labels, boxes, scores):
                if label != -1:
                    x1, y1, x2, y2 = np.int32([coord * image_scale for coord in box])
                    cv2.rectangle(
                        img=image,
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=classes[label][1] if self.classes else [0, 0, 255],
                        thickness=box_thickness
                    )

                    title = f"{classes[label][0]}: {score:.4f}" if self.classes else f"{label}: {score:.4f}"
                    w_text, h_text = cv2.getTextSize(
                        title,
                        cv2.FONT_HERSHEY_PLAIN,
                        font_scale,
                        text_thickness
                    )[0]

                    cv2.rectangle(
                        img=image,
                        pt1=(x1, y1 + int(1.6 * h_text)),
                        pt2=(x1 + w_text, y1),
                        color=classes[label][1] if self.classes else [0, 0, 255],
                        thickness=-1
                    )

                    cv2.putText(
                        img=image,
                        text=title,
                        org=(x1, y1 + int(1.3 * h_text)),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=font_scale,
                        color=(255, 255, 255),
                        thickness=text_thickness,
                        lineType=cv2.LINE_AA
                    )

            cv2.imwrite(save_path, image)

    def compute(self):
        pass

    def started(self, engine):
        self.reset()

    @torch.no_grad()
    def iteration_completed(self, engine):
        output = self.output_transform(engine.state.output)
        self.update(output)

    def completed(self, engine):
        self.compute()

    def _attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
