import torch
import numpy as np
from torch import nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from module import ConvAttention, PreNorm, FeedForward
from TokenLearner import TokenLearnerModule, TransformerBlock
from torchsummary import summary


class Transformer(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class NaiveTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, scale, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth - 1):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TransformerBlock(dim, heads, d_head=dim_head, forward_expansion=scale, dropout=dropout, last_stage=last_stage)),
                PreNorm(dim, FeedForward(dim, dim*scale, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class CvT(nn.Module):
    def __init__(self, image_size, in_channels, num_classes, dim=64, kernels=[4, 3, 3], strides=[2, 2, 2], padding=[1, 1, 1],
                 heads=[1, 2, 4] , depth = [1, 2, 4], pool='cls', dropout=0., emb_dropout=0., scale_dim=4, with_TL=False):
        super().__init__()




        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim
        self.shrink_param = 8  # spatial pooling factor when using TL.
        self.N_tokens     = 16 # How many tokens will you extract from TL?
        self.with_TL      = with_TL # Are you willing to use TL?
        self.N_frames     = 1 # Is this a image? if not, specify number of frames.

        ##### Stage 1 #######
        image_size = image_size // strides[0]
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[0], strides[0], padding[0]),
            Rearrange('b c h w -> b (h w) c', h = image_size, w = image_size),
            nn.LayerNorm(dim)
        )
        self.stage1_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=image_size, depth=depth[0], heads=heads[0], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = image_size, w = image_size)
        )

        ##### Stage 2 #######
        in_channels = dim
        scale = heads[1]//heads[0]
        dim = scale*dim
        image_size = image_size // strides[1]
        self.stage2_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[1], strides[1], padding[1]),
            Rearrange('b c h w -> b (h w) c', h = image_size, w = image_size),
            nn.LayerNorm(dim)
        )
        self.stage2_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=image_size, depth=depth[1], heads=heads[1], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = image_size, w = image_size)
        )

        ##### Stage 3 #######
        in_channels = dim
        scale = heads[2] // heads[1]
        dim = scale * dim
        image_size = image_size // strides[2]
        self.stage3_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[2], strides[2], padding[2]),
            Rearrange('b c h w -> b (h w) c', h=image_size, w=image_size),
            nn.LayerNorm(dim)
        )
        if self.with_TL:
            self.TokenLearner = nn.Sequential(
                Rearrange('(b t) (h w) c -> b t c h w', h=image_size, w=image_size, t=self.N_frames),
                TokenLearnerModule(in_ch=dim, N_tokens=self.N_tokens, H_out=image_size//self.shrink_param,
                                                   W_out=image_size//self.shrink_param, reduction="concat", compression=True)
            )
            self.rearrange_tokens = Rearrange('b s d l w -> b s (d l w)', s=self.N_tokens)
            self.NaiveTransformer = NaiveTransformer(dim=dim, depth=depth[2], heads=heads[2], dim_head=self.dim,
                                                     scale=scale_dim, dropout=dropout, last_stage=True)
        else:
            self.stage3_transformer = nn.Sequential(
                Transformer(dim=dim, img_size=image_size, depth=depth[2], heads=heads[2], dim_head=self.dim,
                                                  mlp_dim=dim * scale_dim, dropout=dropout, last_stage=True),
            )


        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_large = nn.Dropout(emb_dropout)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):

        xs = self.stage1_conv_embed(img)
        xs = self.stage1_transformer(xs)

        xs = self.stage2_conv_embed(xs)
        xs = self.stage2_transformer(xs)

        xs = self.stage3_conv_embed(xs)
        
        if self.with_TL :
            xs = self.TokenLearner(xs)
            b, n, _, _, _ = xs.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
  
            xs = self.rearrange_tokens(xs)
            xs = torch.cat((cls_tokens, xs), dim=1)
    
            xs = self.NaiveTransformer(xs)
            xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0]
            xs = self.mlp_head(xs)

        else:
            b, n, _ = xs.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)       # repeat [() 1 dim] of [1 1 dim] batch_size times.
            xs = torch.cat((cls_tokens, xs), dim=1)
            xs = self.stage3_transformer(xs)
            xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0]
            xs = self.mlp_head(xs)
        return xs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dim = 64
    img = torch.ones([4, 3, img_dim, img_dim])
    model = CvT(img_dim, 3, 1000, pool='mean', with_TL=True)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out = model(img)

    print("Shape of out :", out.shape)  # [B, num_classes]
    #model = model.to(device)
    #print(summary(model, (3,img_dim, img_dim)))
