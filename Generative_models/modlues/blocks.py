#%%
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat, reduce

#%%

def exists(val):
    return val is not None

def _pass(val, f):
    if exists(val):
        return val
    return f() if callable(f) else f #need test(1101)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# def Upsample(in_dim, out_dim = None):
#     _upsample = nn.Upsample(scale_factor = 2, mode = 'bicubic')


class SinusoidalPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_ = self.dim // 2
        embeddings = math.log(10000) / (half_ - 1)
        embeddings = torch.exp(torch.arange(half_, device = device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


#%%
def testfunc(asdf):
    return asdf

if __name__ == "__main__":
    input = torch.randn(4, 16, 256)
    emb_fn = SinusoidalPE(256)
    output = emb_fn(input)
    print(output.shape)