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
* Config: https://github.com/phungpx/efficient_det_pytorch/tree/master/configs/PASCAL
* Training
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/PASCAL/pascal_training.yaml
```
* Testing
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/PASCAL/pascal_testing.yaml
```
* Result

|Model|Parameters|Result|
|:---:|:--------:|:----:|
|EffiDet - D0|3,839,117|<img src="https://user-images.githubusercontent.com/61035926/169470818-7968f3bf-c12c-4503-b45e-0f62071c5622.png" width="600"> |
|EffiDet - D1|-|-|
|EffiDet - D2|-|-|
|EffiDet - D3|-|-|
|EffiDet - D4|-|-|
|EffiDet - D5|-|-|
|EffiDet - D6|-|-|
|EffiDet - D7|-|-|
|EffiDet - D7x|-|-|

## COCO
* Config: https://github.com/phungpx/efficient_det_pytorch/tree/master/configs/COCO/
* Training
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/COCO/coco_training.yaml
```
* Testing
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/COCO/coco_testing.yaml
```
* Result

|Model|Parameters|Result|
|:---:|:--------:|:----:|
|EffiDet - D0|3,874,217|<img src="https://user-images.githubusercontent.com/61035926/169502260-a933b1e0-0eda-4535-a793-662ac7c576fc.png" width="600"> |
|EffiDet - D1|-|-|
|EffiDet - D2|-|-|
|EffiDet - D3|-|-|
|EffiDet - D4|-|-|
|EffiDet - D5|-|-|
|EffiDet - D6|-|-|
|EffiDet - D7|-|-|
|EffiDet - D7x|-|-|

## Birdviews
