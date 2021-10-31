import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange


class TokenExtractor(nn.Module):
    def __init__(self, in_ch, N_tokens, H_out, W_out):
        '''
        :param T:        total number of frames of the input tensor
        :param in_ch:    number of channels in the input tensor
        :param N_tokens: Number of tokens per frame : 'S' in the original paper.
        :param H_out:    Output Height : 1 in the original paper.
        :param W_out:    Output Width  : 1 in the original paper.
        '''
        super(TokenExtractor, self).__init__()

        self.generate_map = nn.Conv2d(in_channels=in_ch, out_channels=N_tokens,
                                      kernel_size=(3,3), stride=(1,1), padding=1)
        self.norm         = nn.Sigmoid()
        self.pool         = nn.AdaptiveMaxPool2d((H_out, W_out))
        self.in_ch        = in_ch

    def forward(self, input_tensor):
        '''
        Basically we assume that the input tensors to be 5D : [BS, T, C, H, W].
        In case of using images, we have to expand its first dimension prior to this layer.

        :param input_tensor: tensor of shape [BS, T, C, H, W]
        :return: tokens of shape [BS, T, S, H_out, W_out]
        '''
        BS, T, C, H, W = input_tensor.shape
        assert C == self.in_ch

        # Rearrange and extract attention maps
        print(input_tensor.shape)
        input_tensor = rearrange(input_tensor, 'B T C H W -> (B T) C H W')  # bind the input with (batch, frame len)
        print("after rearrange : ", input_tensor.shape)
        kernels      = self.generate_map(input_tensor)
        kernels      = self.norm(kernels)                                   # Generate Attention kernels, size of [(BT) S H W]
        print("kernel size : ", kernels.shape)

        Attnetions = []
        for K in kernels.transpose(0,1):
            print(K.shape) # [(BT) H W]
            K = repeat(K, 'BT H W -> BT C H W', C=self.in_ch)
            print(K.shape)
            attn_i = torch.mul(input_tensor, K)
            print("kernelwise attnetion shape : ", attn_i.shape)
            attn_i = self.pool(attn_i)
            Attnetions.append(attn_i)

        Attnetions = torch.stack(Attnetions, dim=1) # Sx[BT C H W] -> [BT S C H W]
        print("Attention block shape : ", Attnetions.shape)

        return Attentions





class TokenFuser(nn.Module):
    def __init__(self,
                 shape_):
        super(TokenFuser, self).__init__()

    def forward(self, input_tensor, tokens):
        pass



if __name__ == '__main__':
    tokenlearner = TokenExtractor(in_ch=3, N_tokens=8, H_out=16, W_out=16)
    X = torch.randn(2,1,3,32,32)
    tokenlearner(X)