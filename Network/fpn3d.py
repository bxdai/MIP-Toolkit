import torch.nn as nn

from resnet3d import BasicBlock


class FPN3D(nn.Module):
    def __init__(self) -> None:
        super(FPN3D,self).__init__()