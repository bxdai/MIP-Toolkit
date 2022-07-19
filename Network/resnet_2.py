import torch
from torchsummary import summary
from torch import nn

# Reference implementations https://github.com/liao2000/ML-Notebook/blob/main/ResNet/ResNet_PyTorch.ipynb
#https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class ResBlock(nn.Module):
    def __init__(self,
    in_channels: int,
    out_channels: int, 
    downsample: bool
    ) -> None:
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1,bias=False)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False)
            self.shortcut = nn.Identity()

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):

        shortcut = self.shortcut(input)

        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        output += shortcut
        output = self.relu(output)
        return output



class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()

        # Downsampling is performed by conv3 1, conv4 1, and conv5 1 with a stride of 2
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4,
                               kernel_size=1, stride=1,bias=False)
        self.conv2 = nn.Conv2d(
            out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1,bias=False)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1,bias=False)

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=2 if self.downsample else 1,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()   

        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))

        input = self.bn3(self.conv3(input))
        input = input + shortcut
        return nn.ReLU()(input)

class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, repeat, useBottleneck=False, outputs=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (
                    i+1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (
                i+1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,),resblock(filters[4], filters[4], downsample=False))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(filters[4], outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        # torch.flatten()
        # https://stackoverflow.com/questions/60115633/pytorch-flatten-doesnt-maintain-batch-size
        input = torch.flatten(input, start_dim=1)
        input = self.fc(input)

        return input

def resnet18():
    return ResNet(3, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=1000)

def resnet34():
    return ResNet(3, ResBlock, [3, 4, 6, 3], useBottleneck=False, outputs=1000)

def resnet50():
    return ResNet(3, ResBottleneckBlock, [
                  3, 4, 6, 3], useBottleneck=True, outputs=1000)

def resnet101():
    return ResNet(3, ResBottleneckBlock, [
                   3, 4, 23, 3], useBottleneck=True, outputs=1000)

def resnet152():
    return ResNet(3, ResBottleneckBlock, [
                   3, 8, 36, 3], useBottleneck=True, outputs=1000)


def testResnet18():
    resnet18 = ResNet(3, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=1000)
    resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    #print(resnet18)
    summary(resnet18, (3, 224, 224))

def testResnet101():
    resnet101 = ResNet(3, ResBottleneckBlock, [
                   3, 4, 23, 3], useBottleneck=True, outputs=1000)
    resnet101.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    summary(resnet101, (3, 224, 224))

if __name__ == "__main__":
    testResnet18()
    testResnet101()
