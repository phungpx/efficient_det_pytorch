# Efficient Det
- paper: https://arxiv.org/pdf/1911.09070.pdf
- reference: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

## 1. download pretrained weight of efficient net:
mkdir checkpoint/efficientnet_pretrained_weight
- efficientnet-b0: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
- efficientnet-b1: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth
- efficientnet-b2: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth
- efficientnet-b3: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth
- efficientnet-b4: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth
- efficientnet-b5: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth
- efficientnet-b6: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth
- efficientnet-b7: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth

## 2. training: 
CUDA_VISIBLE_DEVICES=cuda_indice python -m flame configs/voc2007_training.yaml

## 3. testing:
CUDA_VISIBLE_DEVICES=cuda_indice python -m flame configs/voc2007_testing.yaml
