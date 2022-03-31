# [Efficient Det: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)

## 1. References
[1] EfficientDet - Scalable and Efficient Object Detection: https://arxiv.org/pdf/1911.09070.pdf \
[2] EfficientDet - zylo117 Github: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

## 2. Paper Summaries
### 2.1. Overview
<div align="center">
	<img src="https://user-images.githubusercontent.com/61035926/158733951-2655243d-2f02-4518-8234-a22f56cb915d.png" width="400">
</div>

### 2.2. Architecture
<div align="center">
	<img src="https://user-images.githubusercontent.com/61035926/158735408-8f56d5a9-8756-4b1d-8104-3f41f969574c.png" width="400">
</div>

### 2.3. BiFPN
<div align="center">
	<img src="https://user-images.githubusercontent.com/61035926/158735368-5175c894-eef6-4a0a-af58-466def99af55.png" width="400">
</div>

- Cross-Scale Connections:

- Weighted Feature Fusion:

### 2.4. Compound Scaling
<div align="center">
	<img src="https://user-images.githubusercontent.com/61035926/158735457-ce40b6c4-7ea6-44b3-9071-dfd12db55114.png" width="400">
</div>

- Backbone network
- BiFPN network
- Box/class prediction network
- Input image size

### 2.5. Focal Loss

### 2.6. mAP (mean average precision) \
[1] [Explaination](https://docs.google.com/presentation/d/1OHEU1H1EaBNGDofxkptDZvnNXrhPo2rzbMj7mugHgGs/edit?usp=sharing) \
[2] [Examples](https://colab.research.google.com/drive/1LQWWi0IfUKFEtrJk-oAZcXKlf9hQ7cQ5?usp=sharing) \
[3] [Implementation](https://github.com/phungpx/efficient_det_pytorch/blob/master/flame/handlers/metrics/mean_average_precision/mean_average_precision.py)

## 3. Experiments
- Birdview\
![1112](https://user-images.githubusercontent.com/61035926/158735567-1e66669e-9f11-40b0-9a02-8e4635eefcee.jpg)
![1113](https://user-images.githubusercontent.com/61035926/158735597-bc4a48a8-c4a0-4737-b3aa-a80b9d05faea.jpg)
- VOC2007, 2012
- COCO
- TABLE

## 3. Dataset
### 3.1 Todo
- [x] supporting for COCO 2017, PubLayNet dataset with COCO format.
- [x] supporting for PASCAL VOC 2007, 2012 dataset with XML format.
- [x] supporting for dataset with LABELME format.
- [x] supporting for dataset with ALTHEIA format.

### 3.2 Structure of Configs
```
flame/data
	|
	├── coco_dataset.py
	├── pascal_dataset.py
	├── labelme_dataset.py
	└── altheia_dataset.py

configs/
	|
	├── birdview_vehicles_training.yaml
	├── birdview_vehicles_testing.yaml
	├── coco_training.yaml
	├── coco_testing.yaml
	├── publaynet_training.yaml
	├── publaynet_testing.yaml
	├── pascal_training.yaml
	└── pascal_testing.yaml
```

### 3.3 Download
* COCO Train/Val/Test 2017
```bash
https://cocodataset.org/#download
```

* Pascal VOC 2007
```bash
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
```
```bash
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

* Pascal VOC 2012
```bash
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

* Publaynet
```bash
https://developer.ibm.com/exchanges/data/all/publaynet/
```

### 3.4 Dataset Stats
|ID|Dataset Name|Train|Val|Test|Format|
|:--:|:--------:|:--------:|:--:|:--:|:--:|
1|COCO 2017 |118,287|5,000|-|COCO JSON|
2|Pascal VOC 2007 |5,011|4,952|-|PASCAL XML|
3|Pascal VOC 2012 |1,464|1,449|-|PASCAL XML|
4|PubLayNet |335,703|11,245|11,405|COCO JSON|


## 4. Pretrained Weights
### 4.1 Weights Location of Project
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

### 4.2 EfficientNet Pretrained Weights
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

### 4.3 EfficientDet Pretrained Weights
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

## 5. Usage
### 5.1 Todo
- [x] Applied for many dataset format included coco, pascal, labelme, altheia.
- [x] Applied **imgaug** for augmenting data, dataloader with setting 'num_workers', 'pin_memory', 'drop_last' for optimizing training.
- [x] Rearraged training and testing flow with Ignite Pytorch.
- [x] Refactored **Focal Loss** and **mAP** for training and evaluation.
- [x] Applied **region_predictor** function for visualizing predicted results.
- [ ] Updating FP16 (automatic mixed precision), DDP (DistributedDataParallel) for faster training on GPUs.
- [ ] Updating Tensorboard, Profiler.

### 5.2 Usage
* Training
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/voc2007_training.yaml
```

* Testing
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/voc2007_testing.yaml
```

* Inference
