#%%
import math
import torch 
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange
from inspect import isfunction
from functools import partial


#%%
class Attention(nn.Module):
    '''
    Regular MHA based on https://arxiv.org/abs/1706.03762
    '''
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    '''
    Linear attention from https://arxiv.org/abs/1812.01243
    And implementation reference from https://github.com/lucidrains/linear-attention-transformer
    '''
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class EfficientChannelAttention(nn.Module):
    '''
    Efficient channel attention based on https://arxiv.org/abs/1910.03151

    Inputs:
        channels : Number of channels in the input
        gamma, b : gamma and b parameters of the kernel size calculation
    '''
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()

        # Calculate the kernel size
        k = int(abs((math.log2(channels)/gamma)+(b/gamma)))
        k = k if k % 2 else k + 1

        # Create the convolution layer using the kernel size. 
        self.conv = nn.Conv2d(1, 1, [1, k], padding=[0, k//2], bias=False)

        # Pooling and sigmoid functions
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
    '''
    Inputs:
        x : Image tensor of shape (N, C, L, W)
    Outputs:
        Image tensor of shape (N, C, L, W)
    '''
    def forward(self, x):
        # Pool the input tensor to a (N, C, 1, 1) tensor
        att = self.avgPool(x)

        # Reshape the input tensor to be of shape (N, 1, 1, C)
        att = att.permute(0, 2, 3, 1)

        # Compute the channel attention
        att = self.conv(att)

        # Apply the sigmoid function to the channel attention
        att = self.sigmoid(att)

        # Reshape the input tensor to be of shape (N, C, 1, 1)
        att = att.permute(0, 3, 1, 2)

        # Scale the input by the attention
        return x * att.expand_as(x)


#%%
if __name__ == "__main__":
    # debugging cell
    block_ = EfficientChannelAttention(channels=32)
    print(block_, "\n block parameters : ", block_.conv)