import time
import torch
from memory_profiler import profile
from .efficientdet import EfficientDet


@profile
def inference(model, compound_coef):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.eval().to(device)
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'Num of Parameters Version #D{compound_coef}: {num_params} parameters')

    H = W = 512 + 128 * compound_coef
    tensor = torch.rand(1, 3, H, W).to(device)

    n = 10
    t = []
    for i in range(n):
        t1 = time.time()
        with torch.no_grad():
            _ = model.inference(tensor)
        t2 = time.time()
        t.append(t2 - t1)
    t = t[1:]

    return sum(t) / len(t)


if __name__ == '__main__':
    compound_coef = 0
    model = EfficientDet(
        pretrained_weight=None,
        head_only=False,
        num_classes=90,
        compound_coef=compound_coef,
        backbone_pretrained=False,
        iou_threshold=0.5,
        score_threshold=0.3
    )

    inference_time = inference(model=model, compound_coef=compound_coef)
    print(inference_time)
