#%%
# For realtive import
import sys
sys.path.append('/home/seunghoon/projects/diffusion/Paper_implementation/Generative_models/src/blocks')
sys.path.append('/home/seunghoon/projects/diffusion/Paper_implementation/Generative_models/src/helpers')

import torch
from torch import nn, einsum

#%%
try:
    from blocks.unetBlock import unetBlock
    from blocks.Efficient_Channel_Attention import Efficient_Channel_Attention
    from blocks.Multihead_Attn import Multihead_Attn
except ModuleNotFoundError:
    from ..blocks.unetBlock import unetBlock
    from ..blocks.Efficient_Channel_Attention import Efficient_Channel_Attention
    from ..blocks.Multihead_Attn import Multihead_Attn

class DDXM_UNet(nn.Module):
    # in/out_ch : in and out channel dimension. 3 for the image datas
    # emb_ch : embedding channel dimension
    # ch_mult : channel scaling factor for up/downsample blocks
    # t_dim : vector size for the supplied t vector
    # num_blocks : number of blocks on the up/down path
    # blk_types : how should the residual block be structured
    # c_dim : vector size for the supplied c vectors
    # dropout_rate : rate to apply dropout in the model
    # atn_resolution : resolution of the attention blocks
    def __init__(
            self,
            in_ch:int = 3,
            out_ch:int = 3,
            emb_ch:int = 16,
            ch_mult:int = 2,
            t_dim:int = 64,
            num_blocks:int = 1,
            blk_types:list = ["res", "clsAtn", "chnAtn"],
            c_dim:int = 10,
            dropout_rate:float = 0.1,
            atn_resolution:int = 4
    ):
        super(DDXM_UNet, self).__init__()

        self.c_dim = c_dim

        # Input convolution
        self.in_conv = nn.Conv2d(in_ch, emb_ch, 7, padding=3)

        # Downsampling
        # (N, in_ch, L, W) -> (N, emb_ch^(ch_mult*num_blocks), L/(2^num_blocks), W/(2^num_blocks))
        blocks = []
        cur_ch = emb_ch
        for i in range(1, num_blocks+1):
            blocks.append(unetBlock(cur_ch, emb_ch*(2**(ch_mult*i)), blk_types, t_dim, c_dim, dropoutRate=dropout_rate, atn_resolution=atn_resolution))
            if i != num_blocks+1:
                blocks.append(nn.Conv2d(emb_ch*(2**(ch_mult*i)), emb_ch*(2**(ch_mult*i)), kernel_size=3, stride=2, padding=1))
            cur_ch = emb_ch*(2**(ch_mult*i))
        self.downBlocks = nn.Sequential(
            *blocks
        )

        # Intermediate blocks
        # (N, emb_ch^(ch_mult*num_blocks), L/(2^num_blocks), W/(2^num_blocks))
        # -> (N, emb_ch^(ch_mult*num_blocks), L/(2^num_blocks), W/(2^num_blocks))
        intermediate_ch = cur_ch
        self.intermediate = nn.Sequential(
            unetBlock(intermediate_ch, intermediate_ch, blk_types, t_dim, c_dim, dropoutRate=dropout_rate, atn_resolution=atn_resolution),
            Efficient_Channel_Attention(intermediate_ch),
            unetBlock(intermediate_ch, intermediate_ch, blk_types, t_dim, c_dim, dropoutRate=dropout_rate, atn_resolution=atn_resolution),
        )

        # Upsampling
        # (N, embCh^(chMult*num_blocks), L/(2^num_blocks), W/(2^num_blocks)) -> (N, inCh, L, W)
        blocks = []
        for i in range(num_blocks, -1, -1):
            if i == 0:
                blocks.append(unetBlock(emb_ch*(2**(ch_mult*i)), emb_ch*(2**(ch_mult*i)), blk_types, t_dim, c_dim, dropoutRate=dropout_rate, atn_resolution=atn_resolution))
                blocks.append(unetBlock(emb_ch*(2**(ch_mult*i)), out_ch, blk_types, t_dim, c_dim, dropoutRate=dropout_rate, atn_resolution=atn_resolution))
            else:
                blocks.append(nn.ConvTranspose2d(emb_ch*(2**(ch_mult*(i))), emb_ch*(2**(ch_mult*(i))), kernel_size=4, stride=2, padding=1))
                blocks.append(unetBlock(2*emb_ch*(2**(ch_mult*i)), emb_ch*(2**(ch_mult*(i-1))), blk_types, t_dim, c_dim, dropoutRate=dropout_rate, atn_resolution=atn_resolution))
        self.upBlocks = nn.Sequential(
            *blocks
        )
        
        # Final output block
        self.out = nn.Conv2d(out_ch, out_ch, 7, padding=3)

        # Down/up sample blocks
        self.downSamp = nn.AvgPool2d(2) 
        self.upSamp = nn.Upsample(scale_factor=2)

        # Time embeddings
        self.t_emb = nn.Sequential(
                nn.Linear(t_dim, t_dim),
                nn.GELU(),
                nn.Linear(t_dim, t_dim),
            )
        
    # Input:
    #   X - Tensor of shape (N, in_ch, H, W)
    #   t - Batch of encoded t values for each 
    #       X value of shape (N, t_dim)
    #   c - (optional) Batch of encoded c values
    #       of shape (N, c_dim)
    def forward(self, X, t, c=None):
        # Class embedding assertion
        if type(c) != type(None):
            assert type(self.c_dim) != type(None), "c_dim must be specified when using class information."

        # Encode the time embeddings
        t = self.t_emb(t)
        # Saved residuals to add to the upsampling
        residuals = []
        X = self.in_conv(X)
        
        # Downsample the input with storing outputs to connect to the upsample
        b = 0
        while b < len(self.downBlocks):
            X = self.downBlocks[b](X, t, c)
            residuals.append(X.clone())
            b += 1
            if b < len(self.downBlocks) and type(self.downBlocks[b]) == nn.Conv2d:
                X = self.downBlocks[b](X)
                b += 1
            
        # Reverse the residuals
        residuals = residuals[::-1]
        
        # Intermediate blocks for attending to the class
        for b in self.intermediate:
            try:
                X = b(X, t, c)
            except TypeError:
                X = b(X)
        
        # Send the intermediate batch through the upsampling
        # block to get the original shape
        b = 0
        while b < len(self.upBlocks):
            if b < len(self.upBlocks) and type(self.upBlocks[b]) == nn.ConvTranspose2d:
                X = self.upBlocks[b](X)
                b += 1
            if len(residuals) > 0:
                X = self.upBlocks[b](torch.cat((X, residuals[0]), dim=1), t, c)
            else:
                X = self.upBlocks[b](X, t, c)
            b += 1
            residuals = residuals[1:]
        
        # Send the output through the final block
        # and return the output
        return self.out(X)
# %%
    
if __name__ == "__main__":

    model = DDXM_UNet()
    print(model)
    X_in = torch.randn(4, 3, 256, 256)
    t = torch.randn(4, 64)
    c = torch.randn(4, 10)
    out = model(X_in, t, c)
    print(out.shape)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# params = ", n_params/10**6, " M")
# %%
