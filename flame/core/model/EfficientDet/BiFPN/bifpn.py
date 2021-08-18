import torch
from torch import nn
from typing import List, Tuple, Dict

from .bifpn_layer import BiFPNLayer


class BiFPN(nn.Module):
    def __init__(self, compound_coef: int,
                 backbone_out_channels: Dict[int, List[int]] = {0: [40, 112, 320],
                                                                1: [40, 112, 320],
                                                                2: [48, 120, 352],
                                                                3: [48, 136, 384],
                                                                4: [56, 160, 448],
                                                                5: [64, 176, 512],
                                                                6: [72, 200, 576],
                                                                7: [72, 200, 576],
                                                                8: [80, 224, 640]},
                 W_bifpn: List[int] = [64, 88, 112, 160, 224, 288, 384, 384, 384],
                 D_bifpn: List[int] = [3, 4, 5, 6, 7, 7, 8, 8, 8],
                 onnx_export: bool = False, epsilon: float = 1e-4):
        super(BiFPN, self).__init__()
        bifpn_num_layers = D_bifpn[compound_coef]
        bifpn_out_channels = W_bifpn[compound_coef]

        use_P8 = True if compound_coef > 7 else False
        use_attention = True if compound_coef < 6 else False

        self.bifpn = nn.Sequential(
            *[BiFPNLayer(use_P8=use_P8, epsilon=epsilon, onnx_export=onnx_export,
                         use_attention=use_attention, bifpn_out_channels=bifpn_out_channels,
                         backbone_out_channels=backbone_out_channels[compound_coef],
                         first_time=True if i == 0 else False) for i in range(bifpn_num_layers)]
        )

    def forward(self, feature_maps: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        return self.bifpn(feature_maps)
