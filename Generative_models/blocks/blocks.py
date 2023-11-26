#%%
import math
import torch 
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange
from inspect import isfunction
from functools import partial
#%%


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


#%%
class PositionalEncoding(nn.Module):
    def __init__(
            self,
            emb_dim : int,
            p_drop : float,
            max_len : int = 1000,
            apply_dropout : bool = False,
    ):
        '''
        Args : 
            p_drop : dropout probability
            emb_dim : embedding dimension. must be equal to the transformer emb_dim
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)
        self.apply_dropout = apply_dropout

        pos_encoding = torch.zeros(max_len, emb_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, emb_dim, 2).float() / emb_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(name='pos_encoding', tensor=pos_encoding, persistent=False)
        # persistent = False to exclude this component from the state_dict of the model.

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        """
        Inputs :
            t : Time values of shape (N)
        Outputs : 
            positional_encoding : embeddings of (N, dim)
        """
        positional_encoding = self.pos_encoding[t].squeeze(1)
        if self.apply_dropout:
            return self.dropout(positional_encoding)
        return positional_encoding



#%%
class ConvNext(nn.Sequential):
    '''
    Args : 
        in_Ch - Number of channels the input batch has
        out_Ch - Number of chanels the ouput batch should have
        t_dim - (optional) Number of dimensions in the time input embedding
        c_dim - (optional) Number of dimensions in the class input embedding
        p_drop - Rate to apply dropout to each layer in the block
    '''
    def __init__(self, in_Ch, out_Ch, t_dim=None, c_dim=None, p_drop=0.0):
        super(ConvNext, self).__init__()

        # Implementation found at https://arxiv.org/pdf/2201.03545.pdf
        # It has the following structure:
        #   7x7 conv
        #   Layer Norm
        #   1x1 conv
        #   GELU
        #   1x1 conv
        self.block = nn.Sequential(
            nn.Conv2d(in_Ch, in_Ch, 7, padding=3, groups=in_Ch),
            nn.GroupNorm(in_Ch//4 if in_Ch > 4 else 1, in_Ch),
            nn.Conv2d(in_Ch, in_Ch*2, 1),
            nn.GELU(),
            nn.Conv2d(in_Ch*2, out_Ch, 1),
        )
        self.dropout = nn.Dropout2d(p_drop)

        # Residual path
        self.res = nn.Conv2d(in_Ch, out_Ch, 1) if in_Ch != out_Ch else nn.Identity()

        # Optional time vector applied over the channels
        if t_dim != None:
            self.timeProj = nn.Linear(t_dim, in_Ch)
        else:
            self.timeProj = None

        # Optional class vector applied over the channels
        if c_dim != None:
            self.clsProj = nn.Linear(c_dim, in_Ch)
        else:
            self.clsProj = None

    def forward(self, X, t=None, c=None):
        '''
        Inputs:
            X : Tensor of shape (N, inCh, L, W)
            t : (optional) time vector of shape (N, t_dim)
            c : (optional) class vector of shape (N, c_dim)
        Outputs:
            Tensor of shape (N, outCh, L, W)
        '''
        # Quick t and c check
        if t != None and self.timeProj == None:
            raise RuntimeError("t_dim cannot be None when using time embeddings")
        if c != None and self.clsProj == None:
            raise RuntimeError("c_dim cannot be None when using class embeddings")

        # Residual connection
        res = self.res(X)

        # Main section
        if t == None:
            X = self.block(X)
        else:
            # Initial convolution and dropout
            X = self.block[0](X)
            X = self.block[1](X)
            # Time and class embeddings
            t = self.timeProj(t).unsqueeze(-1).unsqueeze(-1)
            c = self.clsProj(c).unsqueeze(-1).unsqueeze(-1)
            # Combine the class, time, and embedding information
            X = X*t + c
            # Output linear projection
            for b in self.block[2:]:
                X = b(X)

        # Dropout before the residual
        X = self.dropout(X)

        # Connect the residual and main sections
        return X + res





#%%
if __name__ == "__main__":
    # debugging cell
    block_ = ConvNext(in_Ch=3, out_Ch=3, t_dim=None, c_dim=10, p_drop=0.0)
    print(block_, "\n block parameters : ", block_.timeProj, block_.clsProj)