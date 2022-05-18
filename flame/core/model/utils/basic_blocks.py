import math
import torch
from torch import nn
from typing import Union, Tuple


class SeparableConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        use_activation: bool = False,
        onnx_export: bool = False
    ):
        super(SeparableConvBlock, self).__init__()
        self.depthwise_conv = Conv2dStaticSamePadding(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=3, stride=1, bias=False
        )
        self.pointwise_conv = Conv2dStaticSamePadding(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3) if use_batch_norm else nn.Identity()
        if use_activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        else:
            self.swish = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.swish(self.bn(x))

        return x


class Conv2dStaticSamePadding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        groups: int = 1,
        **kwargs
    ):
        super(Conv2dStaticSamePadding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            groups=groups,
            **kwargs
        )

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        extra_W = (math.ceil(W / self.stride[1]) - 1) * self.stride[1] - W + self.kernel_size[1]
        extra_H = (math.ceil(H / self.stride[0]) - 1) * self.stride[0] - H + self.kernel_size[0]

        left = extra_W // 2
        right = extra_W - left
        top = extra_H // 2
        bot = extra_H - top

        x = nn.functional.pad(x, [left, right, top, bot])
        x = self.conv(x)

        return x


class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        *args, **kwargs
    ):
        super(MaxPool2dStaticSamePadding, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, *args, **kwargs)
        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        extra_W = (math.ceil(W / self.stride[1]) - 1) * self.stride[1] - W + self.kernel_size[1]
        extra_H = (math.ceil(H / self.stride[0]) - 1) * self.stride[0] - H + self.kernel_size[0]

        left = extra_W // 2
        right = extra_W - left
        top = extra_H // 2
        bot = extra_H - top

        x = nn.functional.pad(x, [left, right, top, bot])
        x = self.pool(x)

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
