"""
resnet_model.py  –  ResNet-18/34/50 in two flavours

Usage:
    from resnet_model import resnet18, resnet18_sf16

The SF16 variants swap nn.Conv2d and nn.Linear for their Q1.15-quantized
counterparts so that the FORWARD PASS runs entirely in SF16 (Q1.15) while
BACKWARD PASS gradients flow through FP32 master weights via STE.
"""

from typing import List, Optional, Type, Union

import torch
import torch.nn as nn

try:
    from sf16_quantizer import Q115Conv2d, Q115Linear, ste_quantize_q115
    _Q115_AVAILABLE = True
except ImportError:
    _Q115_AVAILABLE = False


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _make_conv3x3(in_ch: int, out_ch: int, stride: int = 1,
                  groups: int = 1, dilation: int = 1,
                  use_q115: bool = False) -> nn.Conv2d:
    klass = Q115Conv2d if use_q115 else nn.Conv2d
    return klass(in_ch, out_ch, kernel_size=3, stride=stride,
                 padding=dilation, groups=groups, bias=False,
                 dilation=dilation)


def _make_conv1x1(in_ch: int, out_ch: int, stride: int = 1,
                  use_q115: bool = False) -> nn.Conv2d:
    klass = Q115Conv2d if use_q115 else nn.Conv2d
    return klass(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


# ---------------------------------------------------------------------------
# BasicBlock  (ResNet-18 / 34)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64,
                 dilation: int = 1, norm_layer=None,
                 use_q115: bool = False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = _make_conv3x3(in_ch, out_ch, stride, use_q115=use_q115)
        self.bn1   = norm_layer(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = _make_conv3x3(out_ch, out_ch, use_q115=use_q115)
        self.bn2   = norm_layer(out_ch)
        self.downsample = downsample
        self.stride = stride
        self.use_q115 = use_q115

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        # Activations are NOT quantized here — in real Q1.15 hardware the
        # multiply-accumulate uses a wider int32 accumulator, so inter-layer
        # activations live outside [-1,1).  Clamping here causes amplitude
        # collapse (~32% saturation per block, nearly binary by layer 4).
        return out


# ---------------------------------------------------------------------------
# Bottleneck  (ResNet-50+)
# ---------------------------------------------------------------------------

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64,
                 dilation: int = 1, norm_layer=None,
                 use_q115: bool = False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_ch * (base_width / 64.0)) * groups

        self.conv1 = _make_conv1x1(in_ch, width, use_q115=use_q115)
        self.bn1   = norm_layer(width)
        self.conv2 = _make_conv3x3(width, width, stride=stride, groups=groups,
                                   dilation=dilation, use_q115=use_q115)
        self.bn2   = norm_layer(width)
        self.conv3 = _make_conv1x1(width, out_ch * self.expansion, use_q115=use_q115)
        self.bn3   = norm_layer(out_ch * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_q115 = use_q115

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        # Same as BasicBlock — no per-block activation quantization.
        return out


# ---------------------------------------------------------------------------
# ResNet backbone
# ---------------------------------------------------------------------------

BlockType = Union[Type[BasicBlock], Type[Bottleneck]]


class ResNet(nn.Module):

    def __init__(self,
                 block: BlockType,
                 layers: List[int],
                 num_classes: int = 10,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer=None,
                 use_q115: bool = False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_q115    = use_q115

        self.in_ch       = 64
        self.dilation    = 1
        self.groups      = groups
        self.base_width  = width_per_group

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element list")

        # Stem – CIFAR-10 friendly (smaller kernel, no maxpool)
        conv_class = Q115Conv2d if use_q115 else nn.Conv2d
        self.conv1   = conv_class(3, self.in_ch, kernel_size=3, stride=1,
                                  padding=1, bias=False)
        self.bn1     = norm_layer(self.in_ch)
        self.relu    = nn.ReLU(inplace=True)
        # (no maxpool – adapted for 32×32 CIFAR images)

        self.layer1  = self._make_layer(block, 64,  layers[0])
        self.layer2  = self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
        self.layer3  = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        self.layer4  = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        fc_class = Q115Linear if use_q115 else nn.Linear
        self.fc  = fc_class(512 * block.expansion, num_classes)

        # Weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Q115Conv2d if use_q115 else nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

        # After init, clamp weights to Q1.15 range
        if use_q115:
            with torch.no_grad():
                for p in self.parameters():
                    p.clamp_(-1.0, 1.0)

    def _make_layer(self, block: BlockType, out_ch: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer  = self._norm_layer
        downsample  = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.in_ch != out_ch * block.expansion:
            if self.use_q115:
                downsample = nn.Sequential(
                    Q115Conv2d(self.in_ch, out_ch * block.expansion,
                               kernel_size=1, stride=stride, bias=False),
                    norm_layer(out_ch * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    _make_conv1x1(self.in_ch, out_ch * block.expansion,
                                  stride=stride),
                    norm_layer(out_ch * block.expansion),
                )

        layers = [
            block(self.in_ch, out_ch, stride, downsample,
                  self.groups, self.base_width, previous_dilation,
                  norm_layer, use_q115=self.use_q115)
        ]
        self.in_ch = out_ch * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_ch, out_ch,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer,
                                use_q115=self.use_q115))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is pre-quantised to Q1.15 by the data pipeline before
        # being passed here – do not re-quantise inside the model.
        x = self.conv1(x)   # weights are Q1.15 via Q115Conv2d
        x = self.bn1(x)
        x = self.relu(x)
        # Stem activations stay in float32 (int32 accumulator equivalent)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)         # weights are Q1.15 via Q115Linear

        # Snap the final logits to Q1.15 so outputs are representable in SF16.
        # We use STE so gradients still flow normally.
        if self.use_q115:
            x = ste_quantize_q115(x)

        return x


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def resnet18(num_classes: int = 10, **kwargs) -> ResNet:
    """Standard FP32 ResNet-18 (CIFAR-10 stem)."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def resnet18_sf16(num_classes: int = 10, **kwargs) -> ResNet:
    """SF16 (Q1.15) ResNet-18 – forward pass in Q1.15, backward in FP32."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                  use_q115=True, **kwargs)


def resnet34(num_classes: int = 10, **kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def resnet34_sf16(num_classes: int = 10, **kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                  use_q115=True, **kwargs)


def resnet50(num_classes: int = 10, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def resnet50_sf16(num_classes: int = 10, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                  use_q115=True, **kwargs)
