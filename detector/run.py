import cv2
import argparse
import numpy as np
# from tqdm import tqdm
from pathlib import Path
# from natsort import natsorted
from typing import Callable, Tuple, Optional

import utils

import os
import sys
sys.path.append(os.environ['PWD'])


def process_video(
    predictor: Callable,
    video_path: str = None,
    # stride: int = 1,  # stride to skip duplicated frames
    frame_size: Optional[Tuple[int, int]] = None,
    FPS: int = 20,  # frame rate of the created video stream
    output_dir: str = None,  # where save processed frames
    show: bool = False,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    video_reader = cv2.VideoCapture(str(video_path))
    # sucess, frame = video_reader.read()
    if frame_size is None:
        frame_width = int(video_reader.get(3))
        frame_height = int(video_reader.get(4))
        frame_size = (frame_width, frame_height)

    if video_path.suffix == '.mp4':
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    elif video_path.suffix == '.avi':
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    else:
        raise ValueError(f'Can not read video with suffix {video_path.suffix}')

    video_writer = cv2.VideoWriter(
        filename=str(output_dir.joinpath(f'{video_path.stem}_output{video_path.suffix}')),
        fourcc=fourcc, fps=FPS, frameSize=frame_size
    )

    while (video_reader.isOpened()):  # loop video to the end of video
        success, frame = video_reader.read()  # get frame by frame with status

        # ret: a boolean indicating if the frame was successfully read or not.
        if success:
            frame = process_image(frame, predictor)
            video_writer.write(frame)

            if show:
                cv2.imshow("FRAME", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):  # close stream video by pressing 'q' button on keyboard
                    cv2.destroyAllWindows()
                    break

        else:
            print('Stream is disconnected.')
            break

    video_reader.release()
    video_writer.release()
    # cv2.destroyAllWindows()


def process_image(image, predictor):
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
    parser.add_argument('--frame-size')
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
            # stride=args.stride,
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
            image = process_image(image, predictor)

            if args.show:
                cv2.imshow(image_path.name, image)
                cv2.waitKey()
                cv2.destroyAllWindows()

            cv2.imwrite(str(output_dir.joinpath(image_path.name)), image)
