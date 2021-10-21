import math
import torch
from torch import nn
from typing import Optional, Union, Tuple
import torch.nn.functional as F


__all__ = [
    'SwishImplementation',
    'MemoryEfficientSwish',
    'Swish',
    'Conv2dStaticSamePadding',
    'MaxPool2dStaticSamePadding',
    'SeparableConvBlock'
]


class SeparableConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        use_batchnorm: bool = True,
        use_activation: bool = False,
        onnx_export: bool = False
    ) -> None:
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise = Conv2dStaticSamePadding(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=3,
            stride=1,
            bias=False
        )

        self.pointwise = Conv2dStaticSamePadding(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )

        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.batchnorm = nn.BatchNorm2d(
                num_features=out_channels,
                momentum=0.01,
                eps=1e-3
            )

        self.use_activation = use_activation
        if self.use_activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''DW -> PW [-> BN -> SWISH]'''

        x = self.depthwise(x)
        x = self.pointwise(x)

        if self.use_batchnorm:
            x = self.batchnorm(x)

        if self.use_activation:
            x = self.swish(x)

        return x


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dStaticSamePadding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        groups: int = 1,
        dilation: int = 1,
        **kwargs
    ) -> None:
        super(Conv2dStaticSamePadding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
            **kwargs
        )

        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)

        return x


class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ) -> None:
        super(MaxPool2dStaticSamePadding, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)

        return x
