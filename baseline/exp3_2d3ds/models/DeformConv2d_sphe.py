import math
import sys

import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from typing import Optional, Tuple
from torchvision.extension import _assert_has_ops



def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Performs Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

    Args:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
            offsets to be applied for each position in the convolution kernel.
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]): convolution weights,
            split into groups of size (in_channels // groups)
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
            masks to be applied for each position in the convolution kernel. Default: None

    Returns:
        Tensor[batch_sz, out_channels, out_h, out_w]: result of convolution

    Examples::
        >>> input = torch.rand(4, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kh, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
        >>> mask = torch.rand(4, kh * kw, 8, 8)
        >>> out = deform_conv2d(input, offset, weight, mask=mask)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([4, 5, 8, 8])
    """

    _assert_has_ops()
    out_channels = weight.shape[0]

    use_mask = mask is not None

    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, in_h, in_w = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            "Got offset.shape[1]={}, while 2 * weight.size[2] * weight.size[3]={}".format(
                offset.shape[1], 2 * weights_h * weights_w))

    return torch.ops.torchvision.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,)




class DeformConv2d_sphe(nn.Module):
    """
    See :func:`deform_conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode = "zeros"
    ):
        super(DeformConv2d_sphe, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = Parameter(torch.empty(out_channels, in_channels // groups,
                                            self.kernel_size[0], self.kernel_size[1]))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # # if self.kernel_size == (8,8):
        # #     self.offset_sphe = self.return_offset_sphe(torch.zeros(1,3,100,200), False, offset_file = '').cuda()
        # # elif self.kernel_size == (4,4):
        # #     self.offset_sphe = self.return_offset_sphe(torch.zeros(1,3,24,49), False, offset_file = '').cuda()
        # # elif self.kernel_size == (3,3):
        # #     self.offset_sphe = self.return_offset_sphe(torch.zeros(1,3,11,23), False, offset_file = '').cuda()
        # if self.kernel_size == (8,8):
        #     self.offset_sphe = self.return_offset_sphe(torch.zeros(1,3,100,100), False, offset_file = '').cuda()
        # elif self.kernel_size == (4,4):
        #     self.offset_sphe = self.return_offset_sphe(torch.zeros(1,3,24,24), False, offset_file = '').cuda()
        # elif self.kernel_size == (3,3):
        #     self.offset_sphe = self.return_offset_sphe(torch.zeros(1,3,11,11), False, offset_file = '').cuda()
        # else:
        #     sys.exit("Not registered DeformConv2d_sphe ", self.kernel_size)
        # self.offset_sphe.require_gradient = False
        
        self.init = 0
             
        self.reset_parameters()


        

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def return_offset_sphe(self, x, isactiv, offset_file = ''):
        h2 = int((x.shape[2]-self.kernel_size[0]+2*self.padding[0])/self.stride[0]+1)
        w2 = int((x.shape[3]-self.kernel_size[1]+2*self.padding[1])/self.stride[1]+1)
        ## https://cs231n.github.io/convolutional-networks/
        if isactiv:
            if offset_file == '':
                offset_file = './models/OFFSETS/offset_'+str(w2)+'_'+str(h2)+'_'+str(self.kernel_size[0])+'_'+str(self.kernel_size[1])+'_'+str(self.stride[0])+'_'+str(self.stride[1])+'_1.pt'                    
                offset = torch.load(offset_file).cuda()
                #offset = torch.cat([offset for _ in range(x.shape[0])],dim=0)
                print("Loading offset file: ", offset_file)
        else:
            offset = torch.zeros(1,2*self.kernel_size[0]*self.kernel_size[1],h2,w2).cuda()
        print("OFFSET Shape ",offset.shape)
        return offset


    def forward(self, input: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
                out_height, out_width]): offsets to be applied for each position in the
                convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width,
                out_height, out_width]): masks to be applied for each position in the
                convolution kernel.
        """
        if self.init == 0:
            print("Using DeformConv2d_sphe")
            print("INPUT Shape ",input.shape)
            self.offset_sphe = self.return_offset_sphe(input, True, offset_file = '').cuda()
            self.offset_sphe.require_gradient = False
            self.init = 1
        offset_sphe_cat = torch.cat([self.offset_sphe for _ in range(input.shape[0])],dim=0).cuda()
        return deform_conv2d(input, offset_sphe_cat, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)



    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)
