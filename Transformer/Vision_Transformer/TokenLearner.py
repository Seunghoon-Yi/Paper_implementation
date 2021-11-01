import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange


class TokenExtractor(nn.Module):
    def __init__(self, in_ch, N_tokens, H_out, W_out, reduction = 'stack'):
        '''
        :param T:        total number of frames of the input tensor
        :param in_ch:    number of channels in the input tensor
        :param N_tokens: Number of tokens per frame : 'S' in the original paper.
        :param H_out:    Output Height : 1 in the original paper.
        :param W_out:    Output Width  : 1 in the original paper.
        :param reduction : 'stack' or 'concat'.
        '''
        super(TokenExtractor, self).__init__()

        self.generate_map = nn.Conv2d(in_channels=in_ch, out_channels=N_tokens,
                                      kernel_size=(3,3), stride=(1,1), padding=1)
        self.norm         = nn.Sigmoid()
        self.pool         = nn.AdaptiveMaxPool2d((H_out, W_out))
        self.in_ch        = in_ch
        self.reduction    = reduction

    def forward(self, input_tensor):
        '''
        Basically we assume that the input tensors to be 5D : [B, T, C, H, W].
        In case of using images, we have to expand its first dimension prior to this layer.

        :param input_tensor: tensor of shape [B, T, C, H, W]
        :return: tokens of shape [B, T, S, H_out, W_out]
        '''
        BS, T, C, H, W = input_tensor.shape
        assert C == self.in_ch
        assert (self.reduction == 'stack') or (self.reduction == 'concat')

        # Rearrange and extract attention maps
        print(input_tensor.shape)
        input_tensor = rearrange(input_tensor, 'B T C H W -> (B T) C H W')  # bind the input with (batch, frame len)
        kernels      = self.generate_map(input_tensor)
        kernels      = self.norm(kernels)                                   # Generate Attention kernels, size of [(BT) S H W]

        Attentions = []
        for K in kernels.transpose(0,1):
            print(K.shape) # [(BT) H W]
            K = repeat(K, '(B T) H W -> (B T) C H W', B=BS, C=self.in_ch)
            attn_i = torch.mul(input_tensor, K)
            attn_i = self.pool(attn_i)
            Attentions.append(attn_i)

        Attentions = torch.stack(Attentions, dim=1) #

        if self.reduction == 'stack':
            Attentions = rearrange(Attentions, '(B T) S C H W -> B T (S C) H W', B=BS, C=self.in_ch)
        elif self.reduction == 'concat':
            Attentions = rearrange(Attentions, '(B T) S C H W -> B (T S) C H W', B=BS, C=self.in_ch)
        print("Attention block shape : ", Attentions.shape, '\n'*3)

        return Attentions


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, device):
        super(MultiHeadAttention, self).__init__()

    def forward(self, x):
        pass

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout, device, forward_expansion = 2):
        super(TransformerBlock, self).__init__()

    def forward(self, x):
        pass



class TokenFuser(nn.Module):
    def __init__(self, C, S, shape_, reduction):
        super(TokenFuser, self).__init__()

        self.C = C
        self.S = S
        self.B, self.T_, self.C_, self.H_, self.W_ = shape_
        self.reduction = reduction
        print("Tokens shape : ", shape_)

        assert (self.reduction == 'stack') or (self.reduction == 'concat')
        self.y_dim  = self.T_*self.C_ // self.C
        self.proj_Y = nn.Linear(self.y_dim, self.y_dim)
        self.proj_X = nn.Linear(self.C, self.H_*self.W_*self.S)
        self.norm   = nn.Sigmoid()


    def forward(self, orig_tensor, tokens):
        if self.reduction == 'stack':
            tokens = rearrange(tokens, 'B T (S C) H W -> B C H W (T S)', S=self.S)
        elif self.reduction == 'concat':
            tokens = rearrange(tokens, 'B (T S) C H W -> B C H W (T S)', S=self.S)
        print("Tokens after rearranged : ", tokens.shape)

        BS, frames, C, H, W = orig_tensor.shape
        print("Origianl Tensor : ", orig_tensor.shape)
        Y = self.proj_Y(tokens)
        Y = rearrange(Y, 'B C H W (T S) -> (B T) (H W S) C', S=self.S)                       # Actually (BT)(H_W_S)C
        print("Y shape : ", Y.shape) ; print("X shape : ", orig_tensor.shape)
        Bw = orig_tensor.permute(0,1,3,4,2)
        Bw = self.norm(self.proj_X(Bw))
        Bw = rearrange(Bw, 'B T H W C -> (B T) (H W) C', C=self.S*self.H_*self.W_, B=self.B) # Actually (BT)(HW)(H_W_S)
        print("weight mtx to Y : ", Bw.shape)
        BwY = torch.bmm(Bw, Y)
        print("after matmul : ", BwY.shape)

        out = rearrange(BwY, '(B T) (H W) C -> B T H W C', B=self.B, T=frames, H=H, W=W, C=C)

        return out



if __name__ == '__main__':

    n_tokens = 8
    in_channels = 32
    mode     = 'stack'

    tokenlearner = TokenExtractor(in_ch=in_channels, N_tokens=n_tokens, H_out=4, W_out=4, reduction=mode)
    X = torch.randn(4,128,in_channels,64,64) # B T C H W
    Attentions = tokenlearner(X)

    repr_tensor = TokenFuser(C=in_channels, S=n_tokens, shape_=Attentions.shape, reduction=mode)
    output = repr_tensor(orig_tensor=X, tokens=Attentions)
    print("final output : ", output.shape)
