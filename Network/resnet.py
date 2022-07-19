from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

#Reference implementations https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

#定义3x3带padding的卷积
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

#定义1x1的卷积
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 用于resnet18和resnet34基本残差结构块
#downsample对应虚线的残差结构
# # Downsampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
class BasicBlock(nn.Module):
    #通道扩充系数，基数是64
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #论文中模型架构的虚线部分，需要下采样
        if self.downsample is not None:
            identity = self.downsample(x)
        
        #shortcut连接
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # pytorch 实现 Bottleneck 是在3x3卷积(self.conv2)的设置stride = 2
    # 原始论文(https://arxiv.org/abs/1512.03385)中实现 Bottleneck 是在1x1卷积(self.conv1)的设置stride = 2
    # 这样做提高了准确率。 这个变体也被称为ResNet V1.5 参考https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    #通道扩充系数
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch(每个残差块最后一个BN用零初始化),
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.(这样每个残差块从零开始，就好像identity)
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677 (精度提高0.2~0.3)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    # 创建conv2_x,conv3_x,conv4_x,conv5_x层
    # channel:conv2/3/4/5对应的各种深度的残差结构主分支上的第一个卷积核的个数/通道数
    #   一个卷积层的残差结构个数
    def _make_layer(
        self,
        #残差块类型：可以是BasicBlock或者Bottleneck
        block: Type[Union[BasicBlock, Bottleneck]],
        #残差快第一个卷积的输入通道
        channels: int,
        #残差块数量
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 对于resnet50/101/152层的结构,第一层为虚线残差,进行下采样
        # 对于resnet18/34层的网络会跳过这个判断,因为输入输出shape一致,无需下采样
        # conv2_x的第一层下采样只需增加channel,不要改变高宽(stride = 1)因为输入输出shape都为64×64
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, channels * block.expansion, stride),
                norm_layer(channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channels, channels, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        #Sequential类来实现简单的顺序连接模型
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        '''正向传播实现函数'''

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #全局的平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # 最后的全连接层
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        '''
        正向传播
        '''
        return self._forward_impl(x)

def resnet18() -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34() -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50() -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3])


def resnet101() -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3])


def resnet152() -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3])

def test_pytorch_resnet():
    from torchsummary import summary
    import torchvision.models as models
    print(models.resnet101(False).cuda())
    summary(models.resnet101(False).cuda(), (3, 224, 224))

if __name__ == '__main__':
    test_pytorch_resnet()
    print("*"*100)
    from torchsummary import summary
    resnet = resnet101()
    resnet.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    summary(resnet,(3, 224, 224))