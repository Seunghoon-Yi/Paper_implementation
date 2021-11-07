import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, einsum
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
        input_tensor = rearrange(input_tensor, 'B T C H W -> (B T) C H W')  # bind the input with (batch, frame len)
        kernels      = self.generate_map(input_tensor)
        kernels      = self.norm(kernels)                                   # Generate Attention kernels, size of [(BT) S H W]

        Attentions = []
        for K in kernels.transpose(0,1):
            K = repeat(K, '(B T) H W -> (B T) C H W', B=BS, C=self.in_ch)
            attn_i = torch.mul(input_tensor, K)
            attn_i = self.pool(attn_i)
            Attentions.append(attn_i)

        Attentions = torch.stack(Attentions, dim=1) #

        if self.reduction == 'stack':
            Attentions = rearrange(Attentions, '(B T) S C H W -> B T (S C) H W', B=BS, C=self.in_ch)
        elif self.reduction == 'concat':
            Attentions = rearrange(Attentions, '(B T) S C H W -> B (T S) C H W', B=BS, C=self.in_ch)

        return Attentions


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_head=64, forward_expansion=4, dropout=0.2, last_stage=False):
        super(TransformerBlock, self).__init__()
        self.last_stage = last_stage
        self.heads      = n_head
        self.scale      = (d_head)**-0.5
        project_out = not (n_head == 1 and d_head == d_model)

        self.to_q = nn.Linear(d_model, d_head*n_head, bias=False)
        self.to_k = nn.Linear(d_model, d_head*n_head, bias=False)
        self.to_v = nn.Linear(d_model, d_head*n_head, bias=False)
        self.WO   = nn.Sequential(nn.Linear(d_model, d_head*n_head),
                                  nn.Dropout(dropout)) if project_out else nn.Identity()
        self.FFNN = nn.Sequential(
            nn.Linear(d_model, d_model*forward_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*forward_expansion, d_model),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        if self.last_stage:
            cla_token = x[:,0]
            x = x[:,1:]
            cls_token = rearrange(cla_token.unsqueeze(1), 'b n (h d) -> b h n d', h=h)

        q = self.to_q(x)
        q = rearrange(q, 'b n (d h) -> b h n d', h=h)
        k = self.to_k(x)
        k = rearrange(k, 'b n (d h) -> b h n d', h=h)
        v = self.to_v(x)
        v = rearrange(v, 'b n (d h) -> b h n d', h=h)

        if self.last_stage:
            q = torch.cat((cls_token, q), dim=2)
            k = torch.cat((cls_token, k), dim=2)
            v = torch.cat((cls_token, v), dim=2)

        dots = einsum('b h m d, b h n d -> b h m n', q, k)*self.scale
        attn = dots.softmax(dim=-1)
        out  = einsum('b h m n, b h n d -> b h n d', attn, v)
        out  = rearrange(out, 'b h n d ->b n (h d)', h=h)
        out = self.WO(out)

        return out


class TokenFuser(nn.Module):
    def __init__(self, C, S, shape_, reduction):
        super(TokenFuser, self).__init__()

        self.C = C
        self.S = S
        self.B, self.T_, self.C_, self.H_, self.W_ = shape_
        self.reduction = reduction

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

        BS, frames, C, H, W = orig_tensor.shape
        Y = self.proj_Y(tokens)
        Y = rearrange(Y, 'B C H W (T S) -> (B T) (H W S) C', S=self.S)                       # Actually (BT)(H_W_S)C

        Bw = orig_tensor.permute(0,1,3,4,2)
        Bw = self.norm(self.proj_X(Bw))
        Bw = rearrange(Bw, 'B T H W C -> (B T) (H W) C', C=self.S*self.H_*self.W_, B=self.B) # Actually (BT)(HW)(H_W_S)
        BwY = torch.bmm(Bw, Y)

        out = rearrange(BwY, '(B T) (H W) C -> B T H W C', B=self.B, T=frames, H=H, W=W, C=C)

        return out


class TokenLearnerModule(nn.Module):
    def __init__(self, in_ch, N_tokens, H_out, W_out, reduction, compression=True):
        super(TokenLearnerModule, self).__init__()

        self.C = in_ch
        self.S = N_tokens
        self.H_out = H_out
        self.W_out = W_out
        self.reduction = reduction
        self.compression = compression

        self.Compressor  = TokenExtractor(in_ch, N_tokens, H_out, W_out, reduction)
        self.Transformer = None
        #self.Fuser       = TokenFuser(in_ch, N_tokens, )

    def forward(self, input_tensor):
        if self.compression :
            return self.Compressor(input_tensor)
        else:
            pass




if __name__ == '__main__':

    n_tokens = 8
    in_channels = 256
    mode     = 'concat'

    tokenlearner = TokenLearnerModule(in_ch=in_channels, N_tokens=n_tokens, H_out=1, W_out=1, reduction=mode)
    X = torch.randn(4,1,in_channels,8,8) # B T C H W
    Attentions = tokenlearner(X)

    repr_tensor = TokenFuser(C=in_channels, S=n_tokens, shape_=Attentions.shape, reduction=mode)
    output = repr_tensor(orig_tensor=X, tokens=Attentions)
    print("final output : ", output.shape)
