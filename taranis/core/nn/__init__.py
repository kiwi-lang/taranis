
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class MaskLayer(nn.Module):
    """Learn a mask to extract features that are most useful
    
    That should promote sparsity as it will cancel some inputs
    Might increase robustness as pixel outside of the main focus will hae less impact
    """
    def __init__(self, size, allow_negative:bool = False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size, **factory_kwargs))
        self.reset_parameters()
        self.allow_negative = allow_negative
        
    def reset_parameters(self) -> None:
        self.weight.data.fill_(0)
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        if not self.allow_negative:
            self.weight.data[self.weight.data < 0] = 0
        return input * self.weight


def LinearIdentity(in_features, out_features):
    """The idea is to help training by forwarding the X by default"""
    layer = nn.Linear(in_features, out_features)
    for i in range(min(in_features, out_features)):
        layer.weight.data[i, i] = 1
    return layer



def simple_conv2d(m, k, **kwargs):
    """  m:        inC x  H x  W
         k: outC x inC x kH x kW """
    
    x = F.conv2d(m.view(1, *m.shape), k, **kwargs)
    n, c, h, w = x.shape
    return x.view(c, h, w)




def conv2d_iwk(images, kernels, **kwargs):
    """ Apply N kernels to a batch of N images

        Images : N x inC x H x W
        Kernels: N x outC x inC x kH x kW
        Output : N x outC x size(Conv2d)
    """

    data = []
    for image, out_kernels in zip(images, kernels):
        val = simple_conv2d(image, out_kernels, **kwargs)

        c, h, w = val.shape

        data.append(val.view(1, c, h, w))

    return torch.cat(data)
