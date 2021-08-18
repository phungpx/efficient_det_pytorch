import torch
from torch import nn
from typing import List, Tuple, Dict
from .utils import MaxPool2dStaticSamePadding, Conv2dStaticSamePadding, SeparableConvBlock
from .utils import MemoryEfficientSwish, Swish


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

        self.bifpn_num_layers = D_bifpn[compound_coef]
        self.bifpn_out_channels = W_bifpn[compound_coef]

        self.epsilon = epsilon
        self.use_P8 = True if compound_coef > 7 else False  # add P8 when using D7x version
        self.use_attention = True if compound_coef < 6 else False  # no use attention for D6, D7, D7x version

        # Conv Layers for P6, P7, P8 after P3, P4, P5 output features of Efficient Net -------------------------------
        '''
            Create Convolution for converting P3, P4, P5 with different out channels to
            P3_in, P4_in, P5_in, P6_in, P7_in with same out_channels (bifpn_out_channels)
            and resolution decrease 2 times when P go deeper.
        '''
        P3_out_channels = backbone_out_channels[compound_coef][0]
        P4_out_channels = backbone_out_channels[compound_coef][1]
        P5_out_channels = backbone_out_channels[compound_coef][2]

        self.P3_to_P3_in_Conv = nn.Sequential(
            Conv2dStaticSamePadding(in_channels=P3_out_channels, out_channels=self.bifpn_out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=self.bifpn_out_channels, momentum=0.01, eps=1e-3)
        )

        self.P4_to_P4_in_Conv = nn.Sequential(
            Conv2dStaticSamePadding(in_channels=P4_out_channels, out_channels=self.bifpn_out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=self.bifpn_out_channels, momentum=0.01, eps=1e-3)
        )

        self.P5_to_P5_in_Conv = nn.Sequential(
            Conv2dStaticSamePadding(in_channels=P5_out_channels, out_channels=self.bifpn_out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=self.bifpn_out_channels, momentum=0.01, eps=1e-3)
        )

        self.P5_to_P6_in_Conv = nn.Sequential(
            Conv2dStaticSamePadding(in_channels=P5_out_channels, out_channels=self.bifpn_out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=self.bifpn_out_channels, momentum=0.01, eps=1e-3),
            MaxPool2dStaticSamePadding(kernel_size=3, stride=2)
        )

        self.P6_in_to_P7_in_Conv = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)

        if self.use_P8:
            self.P7_in_to_P8_in_Conv = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)

        # Conv Layers -----------------------------------------------------------------------------------------------
        # Conv Layers for Top-Down Pathway, using for calculating P6_td, P5_td, P4_td, P3_td
        # Ex: P6_td = td_Conv[(w1 * P6_in + w2 * up_Resize(P7_in)) / (w1 + w2 + epsilon)]
        if self.use_P8:
            self.P7_td_Conv = SeparableConvBlock(in_channels=self.bifpn_out_channels, onnx_export=onnx_export)
        self.P6_td_Conv = SeparableConvBlock(in_channels=self.bifpn_out_channels, onnx_export=onnx_export)
        self.P5_td_Conv = SeparableConvBlock(in_channels=self.bifpn_out_channels, onnx_export=onnx_export)
        self.P4_td_Conv = SeparableConvBlock(in_channels=self.bifpn_out_channels, onnx_export=onnx_export)
        self.P3_td_Conv = SeparableConvBlock(in_channels=self.bifpn_out_channels, onnx_export=onnx_export)

        # Conv Layers for Bottom-Up Pathway, using for P4_out, P5_out, P6_out, P7_out
        # Ex: P6_out = out_Conv[(w1 * P6_in + w2 * P6_td + w3 * down_Resize(P5_out)) / (w1 + w2 + w3 + epsilon)]
        self.P4_out_Conv = SeparableConvBlock(in_channels=self.bifpn_out_channels, onnx_export=onnx_export)
        self.P5_out_Conv = SeparableConvBlock(in_channels=self.bifpn_out_channels, onnx_export=onnx_export)
        self.P6_out_Conv = SeparableConvBlock(in_channels=self.bifpn_out_channels, onnx_export=onnx_export)
        self.P7_out_Conv = SeparableConvBlock(in_channels=self.bifpn_out_channels, onnx_export=onnx_export)
        if self.use_P8:
            self.P8_out_Conv = SeparableConvBlock(in_channels=self.bifpn_out_channels, onnx_export=onnx_export)

        # Resize Layers --------------------------------------------------------------------------------------------
        # Resize Layers (Upsampling Layers) for Top-Down Pathway, using for upsampling P7_in, P6_td, P5_td, P4_td
        # Ex: P6_td = td_Conv[(w1 * P6_in + w2 * up_Resize(P7_in)) / (w1 + w2 + epsilon)]
        if self.use_P8:
            self.P8_Up_Resize = nn.Upsample(scale_factor=2, mode='nearest')
        self.P7_Up_Resize = nn.Upsample(scale_factor=2, mode='nearest')
        self.P6_Up_Resize = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_Up_Resize = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_Up_Resize = nn.Upsample(scale_factor=2, mode='nearest')

        # Resize Layers (Downsampling Layers) for Bottom-Up Pathway, using for upsampling P3_td, P4_out, P5_out, P6_out
        # Ex: P6_out = out_Conv[(w1 * P6_in + w2 * P6_td + w3 * down_Resize(P5_out)) / (w1 + w2 + w3 + epsilon)]
        self.P3_Down_Resize = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)
        self.P4_Down_Resize = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)
        self.P5_Down_Resize = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)
        self.P6_Down_Resize = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)
        if self.use_P8:
            self.P7_Down_Resize = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)

        # Weight ---------------------------------------------------------------------------------------------------
        # top-down pathway
        # Ex: P6_td = td_Conv[(w1 * P6_in + w2 * up_Resize(P7_in)) / (w1 + w2 + epsilon)]
        self.P6_W1 = nn.Parameter(data=torch.ones(size=(2, ), dtype=torch.float32), requires_grad=True)
        self.P6_W1_ReLU = nn.ReLU()
        self.P5_W1 = nn.Parameter(data=torch.ones(size=(2, ), dtype=torch.float32), requires_grad=True)
        self.P5_W1_ReLU = nn.ReLU()
        self.P4_W1 = nn.Parameter(data=torch.ones(size=(2, ), dtype=torch.float32), requires_grad=True)
        self.P4_W1_ReLU = nn.ReLU()
        self.P3_W1 = nn.Parameter(data=torch.ones(size=(2, ), dtype=torch.float32), requires_grad=True)
        self.P3_W1_ReLU = nn.ReLU()

        # bottom-up pathway
        # Ex: P6_out = out_Conv[(w1 * P6_in + w2 * P6_td + w3 * down_Resize(P5_out)) / (w1 + w2 + w3 + epsilon)]
        self.P4_W2 = nn.Parameter(data=torch.ones(size=(3, ), dtype=torch.float32), requires_grad=True)
        self.P4_W2_ReLU = nn.ReLU()
        self.P5_W2 = nn.Parameter(data=torch.ones(size=(3, ), dtype=torch.float32), requires_grad=True)
        self.P5_W2_ReLU = nn.ReLU()
        self.P6_W2 = nn.Parameter(data=torch.ones(size=(3, ), dtype=torch.float32), requires_grad=True)
        self.P6_W2_ReLU = nn.ReLU()
        self.P7_W2 = nn.Parameter(data=torch.ones(size=(2, ), dtype=torch.float32), requires_grad=True)
        self.P7_W2_ReLU = nn.ReLU()

        # activation
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        P3, P4, P5 = inputs

        # P3_in, P4_in, P5_in, P6_in, P7_in, P8_in
        P3_in = self.P3_to_P3_in_Conv(P3)
        P4_in = self.P4_to_P4_in_Conv(P4)
        P5_in = self.P5_to_P5_in_Conv(P5)
        P6_in = self.P5_to_P6_in_Conv(P5)
        P7_in = self.P6_in_to_P7_in_Conv(P6_in)
        if self.use_P8:
            P8_in = self.P7_in_to_P8_in_Conv(P7_in)
            feature_fusion = (P3_in, P4_in, P5_in, P6_in, P7_in, P8_in)
        else:
            feature_fusion = (P3_in, P4_in, P5_in, P6_in, P7_in)

        if self.use_attention and (not self.use_P8):
            for _ in range(self.bifpn_num_layers):
                feature_fusion = self._fast_normalized_weighted_fusion(inputs=feature_fusion)

        if self.use_P8 or (not self.use_attention):
            for _ in range(self.bifpn_num_layers):
                feature_fusion = self._normal_fusion(inputs=feature_fusion)

        return feature_fusion

    def _fast_normalized_weighted_fusion(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        P3_in, P4_in, P5_in, P6_in, P7_in = inputs

        # Top-Down PathWay: P7_td, P6_td, P5_td, P4_td, P3_td -------------------------------------------------------
        # P6_td = td_Conv[(w1 * P6_in + w2 * up_Resize(P7_in)) / (w1 + w2 + epsilon)]
        P6_W1 = self.P6_W1_ReLU(self.P6_W1)
        W6 = P6_W1 / (P6_W1.sum(dim=0) + self.epsilon)
        P6_td = self.P6_td_Conv(self.swish(W6[0] * P6_in + W6[1] * self.P7_Up_Resize(P7_in)))

        # P5_td = td_Conv[(w1 * P5_in + w2 * up_Resize(P6_td)) / (w1 + w2 + epsilon)]
        P5_W1 = self.P5_W1_ReLU(self.P5_W1)
        W5 = P5_W1 / (P5_W1.sum(dim=0) + self.epsilon)
        P5_td = self.P5_td_Conv(self.swish(W5[0] * P5_in + W5[1] * self.P6_Up_Resize(P6_td)))

        # P4_td = td_Conv[(w1 * P4_in + w2 * up_Resize(P5_td)) / (w1 + w2 + epsilon)]
        P4_W1 = self.P4_W1_ReLU(self.P4_W1)
        W4 = P4_W1 / (P4_W1.sum(dim=0) + self.epsilon)
        P4_td = self.P4_td_Conv(self.swish(W4[0] * P4_in + W4[1] * self.P5_Up_Resize(P5_td)))

        # P3_td = td_Conv[(w1 * P3_in + w2 * up_Resize(P4_td)) / (w1 + w2 + epsilon)]
        P3_W1 = self.P3_W1_ReLU(self.P3_W1)
        W3 = P3_W1 / (P3_W1.sum(dim=0) + self.epsilon)
        P3_td = self.P3_td_Conv(self.swish(W3[0] * P3_in + W3[1] * self.P4_Up_Resize(P4_td)))

        # Bottom-Up Pathway: P7_out, P6_out, P5_out, P4_out, P3_out ------------------------------------------------
        # P3_out = td_Conv[(w1 * P3_in + w2 * up_Resize(P4_td)) / (w1 + w2 + epsilon)]
        P3_out = P3_td

        # P4_out = out_Conv[(w1 * P4_in + w2 * P4_td + w3 * down_Resize(P3_out)) / (w1 + w2 + w3 + epsilon)]
        P4_W2 = self.P4_W2_ReLU(self.P4_W2)
        W4 = P4_W2 / (torch.sum(P4_W2, dim=0) + self.epsilon)
        P4_out = self.P4_out_Conv(self.swish(W4[0] * P4_in + W4[1] * P4_td + W4[2] * self.P3_Down_Resize(P3_out)))

        # P5_out = out_Conv[(w1 * P5_in + w2 * P5_td + w3 * down_Resize(P4_out)) / (w1 + w2 + w3 + epsilon)]
        P5_W2 = self.P5_W2_ReLU(self.P5_W2)
        W5 = P5_W2 / (torch.sum(P5_W2, dim=0) + self.epsilon)
        P5_out = self.P5_out_Conv(self.swish(W5[0] * P5_in + W5[1] * P5_td + W5[2] * self.P4_Down_Resize(P4_out)))

        # P6_out = out_Conv[(w1 * P6_in + w2 * P6_td + w3 * down_Resize(P5_out)) / (w1 + w2 + w3 + epsilon)]
        P6_W2 = self.P6_W2_ReLU(self.P6_W2)
        W6 = P6_W2 / (torch.sum(P6_W2, dim=0) + self.epsilon)
        P6_out = self.P6_out_Conv(self.swish(W6[0] * P6_in + W6[1] * P6_td + W6[2] * self.P5_Down_Resize(P5_out)))

        # P7_out = out_Conv[(w1 * P7_in + w2 * down_Resize(P6_out)) / (w1 + w2 + epsilon)]
        P7_W2 = self.P7_W2_ReLU(self.P7_W2)
        W7 = P7_W2 / (torch.sum(P7_W2, dim=0) + self.epsilon)
        P7_out = self.P7_out_Conv(self.swish(W7[0] * P7_in + W7[1] * self.P6_Down_Resize(P6_out)))

        return P3_out, P4_out, P5_out, P6_out, P7_out

    def _normal_fusion(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        if self.use_P8:
            P3_in, P4_in, P5_in, P6_in, P7_in, P8_in = inputs
        else:
            P3_in, P4_in, P5_in, P6_in, P7_in = inputs

        # Top-Down PathWay: P7_td, P6_td, P5_td, P4_td, P3_td -------------------------------------------------------
        if self.use_P8:
            # P7_td = td_Conv(P7_in + up_Resize(P8_in))
            P7_td = self.P7_td_Conv(self.swish(P7_in + self.P8_Up_Resize(P8_in)))
            # P6_td = td_Conv(P6_in + up_Resize(P7_td))
            P6_td = self.P6_td_Conv(self.swish(P6_in + self.P7_Up_Resize(P7_td)))
        else:
            P6_td = self.P6_td_Conv(self.swish(P6_in + self.P7_Up_Resize(P7_in)))

        P5_td = self.P5_td_Conv(self.swish(P5_in + self.P6_Up_Resize(P6_td)))
        P4_td = self.P4_td_Conv(self.swish(P4_in + self.P5_Up_Resize(P5_td)))
        P3_td = self.P3_td_Conv(self.swish(P3_in + self.P4_Up_Resize(P4_td)))

        # Bottom-Up Pathway: P3_out, P4_out, P5_out, P6_out, P7_out, P8_out ----------------------------------------
        P3_out = P3_td
        P4_out = self.P4_out_Conv(self.swish(P4_in + P4_td + self.P3_Down_Resize(P3_out)))
        P5_out = self.P5_out_Conv(self.swish(P5_in + P5_td + self.P4_Down_Resize(P4_out)))
        P6_out = self.P6_out_Conv(self.swish(P6_in + P6_td + self.P5_Down_Resize(P5_out)))

        if self.use_P8:
            P7_out = self.P7_out_Conv(self.swish(P7_in + P7_td + self.P6_Down_Resize(P6_out)))
            P8_out = self.conv8_down(self.swish(P8_in + self.P7_Down_Resize(P7_out)))
            return P3_out, P4_out, P5_out, P6_out, P7_out, P8_out
        else:
            P7_out = self.P7_out_Conv(self.swish(P7_in + self.P6_Down_Resize(P6_out)))
            return P3_out, P4_out, P5_out, P6_out, P7_out
