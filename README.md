# [Efficient Det: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)

## 1. References
[1] EfficientDet - Scalable and Efficient Object Detection: https://arxiv.org/pdf/1911.09070.pdf \
[2] EfficientDet - zylo117 Github: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

## 2. Dataset
### 2.1 Todo
- [x] suport for COCO 2017 dataset format.
- [x] supporting for PASCAL VOC 2007, 2012 dataset format.
- [x] supporting for LABELME dataset format.
- [x] supporting for ALTHEIA dataset format.

### 2.2 Structure of Configs
```
flame/data
	|
	├── coco_dataset.py
	├── pascal_dataset.py
	├── labelme_dataset.py
	└── altheia_dataset.py

configs/
	|
	├── coco_training.yaml
	├── coco_testing.yaml
	├── pascal_training.yaml
	└── pascal_testing.yaml
```

### 2.3 Download
* COCO Train/Val/Test 2017
```bash
https://cocodataset.org/#download
```

* Pascal VOC 2007
```bash
https://www.kaggle.com/zaraks/pascal-voc-2007
```

* Pascal VOC 2012
```bash
https://www.kaggle.com/huanghanchina/pascal-voc-2012
```

### 2.4 Dataset Stats
|ID|Dataset Name|Train|Val|Test|
|:--:|:--------:|:--------:|:--:|:--:|
1|COCO 2017 |118,287|5,000|-|
2|Pascal VOC 2007 |5,011|4,952|-|
3|Pascal VOC 2012 |1,464|1,449|-|


## 3. Pretrained Weights
### 3.1 Weights Location of Project
* Create Weight Folder
```python
mkdir checkpoint/efficientnet_pretrained_weight
mkdir checkpoint/efficientdet_pretrained_weight
```
* Download Weight
```bash
!wget <weight-path> -O checkpoint/efficientdet_pretrained_weight/<weight-path-name>
```
```bash
!wget <weight-path> -O checkpoint/efficientnet_pretrained_weight/<weight-path-name>
```

### 3.2 EfficientNet Pretrained Weights
```bash
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth
```

### 3.3 EfficientDet Pretrained Weights
```bash
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d0.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d1.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d2.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d3.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d4.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d5.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d6.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d7.pth
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d8.pth
```

## 4. Usage
### 4.1 Todo
[x] Applied for many dataset format included coco, pascal, labelme, altheia.
[x] Rearraged training and testing flow with Ignite Pytorch.
[x] Refactored Focal Loss and mAP for training and evaluation.
[x] Applied *region_predictor* function for visualizing predicted results.
[x] Applied 'lr_scheduler', 'early stopping', dataloader with setting 'num_workers', 'pin_memory', 'drop_last' for optimizing training.
[] Updating FP16 (automatic mixed precision), DDP (DistributedDataParallel) for faster training on GPUs.
[] Updating Tensorboard, Profiler.

### 4.2 Usage
* Training
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/voc2007_training.yaml
```

* Testing
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/voc2007_testing.yaml
```

* mean average precision \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LQWWi0IfUKFEtrJk-oAZcXKlf9hQ7cQ5?usp=sharing)

* inference \
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1n4QoUcpv3wz6lXsWJSBAbRk4ZdO6NnEb/view?usp=sharing)

## 5. Performance
<Updating>
## 6. Explaination
<Updating>
