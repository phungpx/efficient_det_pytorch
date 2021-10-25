import cv2
import argparse
import numpy as np
from pathlib import Path

import utils

import os
import sys
sys.path.append(os.environ['PWD'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='detector/config.yaml')
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--pattern', type=str)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output-dir', default='detector/output/')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    image_paths = list(Path(args.input_dir).glob(args.pattern)) if args.pattern else [Path(args.input_dir)]

    config = utils.load_yaml(args.config)
    predictor = utils.eval_config(config)

    for i, image_path in enumerate(image_paths, 1):
        print('**' * 30)
        print(f'{i} / {len(image_paths)} - {image_path.name}')

        image = cv2.imread(str(image_path))
        prediction = predictor([image])[0]

        boxes = prediction['boxes'].cpu().numpy().astype(np.int32)
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        class_names = prediction['names']
        font_scale = max(image.shape) / 1200
        box_thickness = max(image.shape) // 400
        text_thickness = max(image.shape) // 600

        for (label, class_name, box, score) in zip(labels, class_names, boxes, scores):
            if label != -1:
                x1, y1, x2, y2 = box
                color = (
                    np.random.randint(200, 255),
                    np.random.randint(50, 200),
                    np.random.randint(0, 150)
                )

                cv2.rectangle(
                    img=image,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=color,
                    thickness=box_thickness
                )

                title = f"{class_name}: {score:.4f}"
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
                    color=color,
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

            if args.show:
                cv2.imshow(image_path.name, image)
                cv2.waitKey()
                cv2.destroyAllWindows()

        cv2.imwrite(str(output_dir.joinpath(image_path.name)), image)
