#%%
import math
import torch 
from torch import nn

#%%
class convNext(nn.Sequential):
    '''
    Args : 
        in_Ch - Number of channels the input batch has
        out_Ch - Number of chanels the ouput batch should have
        t_dim - (optional) Number of dimensions in the time input embedding
        c_dim - (optional) Number of dimensions in the class input embedding
        p_drop - Rate to apply dropout to each layer in the block
    '''
    def __init__(self, in_Ch, out_Ch, t_dim=None, c_dim=None, p_drop=0.0):
        super(convNext, self).__init__()

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


