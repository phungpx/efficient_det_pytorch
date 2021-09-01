# [Efficient Det](https://arxiv.org/pdf/1911.09070.pdf)

## 0. References
- EfficientDet(paper): https://arxiv.org/pdf/1911.09070.pdf
- EfficientDet(Github): https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

## 1. Download
> Weights Location
```python
mkdir checkpoint/efficientnet_pretrained_weight
mkdir checkpoint/efficientdet_pretrained_weight
```

> EfficientNet Pretrained Weight
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

> EfficientDet Pretrained Weight


## 2. Usage
> Training
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/voc2007_training.yaml
```

> Testing
```python
CUDA_VISIBLE_DEVICES=<cuda_indice> python -m flame configs/voc2007_testing.yaml
```

> mAP testing
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LQWWi0IfUKFEtrJk-oAZcXKlf9hQ7cQ5?usp=sharing)

> Inference
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1n4QoUcpv3wz6lXsWJSBAbRk4ZdO6NnEb/view?usp=sharing)
