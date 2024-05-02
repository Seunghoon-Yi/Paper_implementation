#%%
import torch
import functools
import numpy as np
from torch import nn
from einops import rearrange
from functools import partial
from Blocks import PositionalEncoding, ConvNext, PreNorm, default, exists, Residual, Upsample, Downsample
from Attn import Attention, LinearAttention, EfficientChannelAttention

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

#%%
class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]
  
  
class Unet(nn.Module):
    def __init__(
        self,
        init_dim=None,
        next_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        use_random_gaussian=False,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, next_dim//8)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: next_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(init_dim, dims, in_out)
        
        if use_convnext:
            block_class = partial(ConvNext, mult=convnext_mult)
        else:
            block_class = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = next_dim * 4
            self.time_mlp = None
            if not use_random_gaussian:
                self.time_mlp = nn.Sequential(
                    PositionalEncoding(next_dim, p_drop=0.0),
                    nn.Linear(next_dim, time_dim),
                    nn.GELU(),
                    nn.Linear(time_dim, time_dim),
                )
            else:
               self.time_mlp = nn.Sequential(
                    GaussianFourierProjection(embed_dim=next_dim),
                    nn.Linear(next_dim, time_dim),
                    nn.GELU(),
                    nn.Linear(time_dim, time_dim),
               )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            print(dim_in, dim_out, is_last)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_class(dim_in, dim_out, t_dim=time_dim),
                        block_class(dim_out, dim_out, t_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        print("mid_dim : ", dims[-1])
        mid_dim = dims[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim, t_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_class(mid_dim, mid_dim, t_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            print(dim_in, dim_out, is_last)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_class(dim_out * 2, dim_in, t_dim=time_dim),
                        block_class(dim_in, dim_in, t_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_class(next_dim, next_dim), nn.Conv2d(next_dim, out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        print("x and t : ", x.shape, t.shape)

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
#%%
# Set up the SDE

def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
  t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=device)
  
sigma =  25.0#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

#%%

if __name__ == "__main__":
    image_size = 256
    channels = 3
    batchSize = 4
    model = Unet(
    init_dim=16,
    next_dim=64,
    channels=channels,
    dim_mults=(1, 2, 4,),
    use_random_gaussian=True,
    )
    #print(model)
    out_feature = model(torch.rand(batchSize, channels, 256, 256), time=torch.Tensor([32]))
# %%
