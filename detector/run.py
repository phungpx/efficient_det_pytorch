import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Callable, Tuple

import utils

import os
import sys
sys.path.append(os.environ['PWD'])


def make_video(
    image_dir: str = None,
    image_extent: str = '.*',
    frame_size: Tuple[int, int] = (960, 540),
    FPS: int = 32,  # fps mong muốn khi ghi
    output_dir: str = None,
):
    '''write a video from all frame images in folder
    '''
    video_writer = cv2.VideoWriter(
        filename=output_dir, fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
        fps=FPS, frameSize=frame_size
    )

    image_paths = list(Path(image_dir).glob(f'**/*{image_extent}'))
    image_paths = natsorted(image_paths, key=lambda x: x.stem)

    for image_path in tqdm(image_paths):
        image = cv2.imread(str(image_path))

        image_path.unlink(missing_ok=False)

        if image is None:
            continue

        video_writer.write(image)

    video_writer.release()


def process_video(
    predictor: Callable,
    video_path: str = None,
    stride: int = 1,  # stride để đọc các frame
    image_extent: str = '.jpg',
    frame_size: Tuple[int, int] = (960, 540),
    FPS: int = 32,  # fps mong muốn khi ghi
    output_dir: str = None  # where save processed frames
):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    count = 0
    video_capture = cv2.VideoCapture(video_path)

    while (video_capture.isOpened()):  # lặp đến hết video
        ret, frame = video_capture.read()  # lấy ra từng frame

        if count % stride != 0:
            continue

        count += 1

        # ret: a boolean indicating if the frame was successfully read or not.
        if ret:
            frame = predict_image(frame, predictor)
            frame_path = output_dir.joinpath(f'{Path(video_path).stem}_{count}{image_extent}')
            cv2.imwrite(str(frame_path), frame)

        # tắt chương trình bằng nút 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    make_video(
        image_dir=output_dir, image_extent=image_extent,
        frame_size=frame_size, FPS=FPS, output_dir=output_dir,
    )


def predict_image(image, predictor):
    prediction = predictor([image])[0]

    boxes = prediction['boxes'].cpu().numpy().astype(np.int32)
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    class_names = prediction['names']
    font_scale = max(image.shape) / 1200
    box_thickness = max(image.shape) // 400
    text_thickness = max(image.shape) // 600

    for (label, class_name, box, score) in zip(labels, class_names, boxes, scores):
        if label == -1:
            continue

        x1, y1, x2, y2 = box
        color = (
            np.random.randint(200, 255),
            np.random.randint(50, 200),
            np.random.randint(0, 150)
        )

        cv2.rectangle(
            img=image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=box_thickness
        )

        title = f"{class_name}: {score:.4f}"
        w_text, h_text = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_PLAIN, font_scale, text_thickness
        )[0]

        cv2.rectangle(
            img=image, pt1=(x1, y1 + int(1.6 * h_text)), pt2=(x1 + w_text, y1),
            color=color, thickness=-1
        )

        cv2.putText(
            img=image, text=title, org=(x1, y1 + int(1.3 * h_text)),
            fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale,
            color=(255, 255, 255), thickness=text_thickness, lineType=cv2.LINE_AA
        )

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='detector/config.yaml')
    # for visualizing image
    parser.add_argument('--image-dir', type=str)
    parser.add_argument('--pattern', type=str)
    parser.add_argument('--show', action='store_true')
    # for visualizing video
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--fps', type=int, default=32)
    parser.add_argument('--frame-size', type=tuple, default=(960, 540))
    parser.add_argument('--stride', type=int, default=1)
    # save video / image
    parser.add_argument('--output-dir', default='detector/output/')
    args = parser.parse_args()

    config = utils.load_yaml(args.config)
    predictor = utils.eval_config(config)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if args.video_path:
        process_video(
            predictor=predictor,
            video_path=args.video_path,
            stride=args.stride,
            frame_size=args.frame_size,
            FPS=args.fps,
            output_dir=args.output_dir,
        )

    if args.image_dir:
        image_paths = list(Path(args.image_dir).glob(args.pattern)) if args.pattern else [Path(args.image_dir)]

        for i, image_path in enumerate(image_paths, 1):
            print('**' * 30)
            print(f'{i} / {len(image_paths)} - {image_path.name}')

            image = cv2.imread(str(image_path))
            image = predict_image(image, predictor)

            if args.show:
                cv2.imshow(image_path.name, image)
                cv2.waitKey()
                cv2.destroyAllWindows()

            cv2.imwrite(str(output_dir.joinpath(image_path.name)), image)
