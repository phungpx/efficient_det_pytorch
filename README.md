# [Efficient Det: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)

[1] EfficientDet - Scalable and Efficient Object Detection: https://arxiv.org/pdf/1911.09070.pdf \
[2] EfficientDet - zylo117 Github: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch \
[3] EfficinetNet - Github: https://github.com/lukemelas/EfficientNet-PyTorch

# TODO
- [x] Dataset format including coco, pascal, labelme and altheia.
- [x] Efficient Det training, inference flow.
- [ ] Augmentations: Mosaic, Mixup, CutMix,...
- [ ] BBox IOU Loss (GIoU, DIoU, CIoU, ...)
* [Distance-IoU Loss](https://arxiv.org/pdf/1911.08287.pdf)
* [Complete-IoU Loss](https://arxiv.org/pdf/2005.03572.pdf)
- [ ] [Using soft-nms instead of normal NMS.](https://arxiv.org/pdf/1704.04503.pdf)
- [ ] FP16 (automatic mixed precision), DDP (DistributedDataParallel) for faster training on GPUs.
- [ ] Tensorboard, Profiler.

# MAIN FUNCTIONS
* [Backbone using all variants of Efficient NetV1](https://github.com/phungpx/efficient_det_pytorch/blob/master/flame/core/model/backbone/__init__.py)
* [BiFPN](https://github.com/phungpx/efficient_det_pytorch/blob/master/flame/core/model/bifpn.py)
* [Regression and Classification Head](https://github.com/phungpx/efficient_det_pytorch/blob/master/flame/core/model/head.py)
* [Anchor Generation](https://github.com/phungpx/efficient_det_pytorch/blob/master/flame/core/model/anchor_generator.py)
* [Efficient Det: combinating all parts together including Efficient NetV1 (backbone), BiFPN (neck), Regressor&Classifier(head) and Anchors](https://github.com/phungpx/efficient_det_pytorch/blob/master/flame/core/model/efficient_det.py)
* [Focal Loss](https://github.com/phungpx/efficient_det_pytorch/blob/master/flame/core/loss/focal_loss.py)
* [COCO Eval](https://github.com/phungpx/efficient_det_pytorch/blob/master/flame/core/metric/COCO_eval.py)
* [mAP](https://github.com/phungpx/efficient_det_pytorch/blob/master/flame/core/metric/mAP.py)
* [Visualization for predicting results](https://github.com/phungpx/efficient_det_pytorch/blob/master/flame/handlers/region_predictor.py)

# DATASET
|ID|Dataset Name|Train|Val|Test|Format|
|:--:|:--------:|:--------:|:--:|:--:|:--:|
1|COCO 2017 |118,287|5,000|-|COCO JSON|
2|Pascal VOC 2007 |5,011|4,952|-|PASCAL XML|
3|Pascal VOC 2012 |1,464|1,449|-|PASCAL XML|
4|PubLayNet |335,703|11,245|11,405|COCO JSON|

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

# USAGE
## VOC 2007, 2012
### Config: https://github.com/phungpx/efficient_det_pytorch/tree/master/configs/PASCAL
### Training
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/PASCAL/pascal_training.yaml
```
### Testing
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/PASCAL/pascal_testing.yaml
```
### Result
* EfficientDet - D0
<div align="center">
	<img src="https://user-images.githubusercontent.com/61035926/169444903-890a341f-2909-40d2-9b7c-3d1c8f36e7c7.png", width="1200">
</div>

* EfficientDet - D1

* EfficientDet - D2

* EfficientDet - D3

* EfficientDet - D4

* EfficientDet - D5

* EfficientDet - D6

* EfficientDet - D7

## COCO

## Birdviews
