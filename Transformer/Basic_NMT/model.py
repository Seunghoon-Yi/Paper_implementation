import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        self.d_K = d_model//n_head
        self.n_head = n_head
        self.d_model = d_model
        # To make sure that model dim = multiplier of heads
        #print(self.d_K, self.n_head, self.d_model)
        assert (self.n_head*self.d_K == self.d_model), "Embed size needs to be divisible by heads"
        # [d_model, [d_K,...,d_K]] 이렇게 d_K씩 다른 representation 익힘.
        self.WQ = nn.Linear(self.d_model, self.d_model, bias=False)
        self.WK = nn.Linear(self.d_model, self.d_model, bias=False)
        self.WV = nn.Linear(self.d_model, self.d_model, bias=False)
        # The W0
        self.fc_out = nn.Linear(self.d_K*self.n_head, d_model)

    def forward(self, Q_ipt, K_ipt, V_ipt, Mask):
        BS = Q_ipt.shape[0]
        l_Q, l_K, l_V = Q_ipt.shape[1], K_ipt.shape[1], V_ipt.shape[1]
        Q = self.WQ(Q_ipt)
        K = self.WK(K_ipt)
        V = self.WV(V_ipt) # (BS, seq_len, d_model)

        # Split inputs to n_heads
        Queries = Q.reshape(BS, l_Q, self.n_head, self.d_K)
        Keys    = K.reshape(BS, l_K, self.n_head, self.d_K)
        Values  = V.reshape(BS, l_V, self.n_head, self.d_K)
        #print(Queries.shape, Keys.shape, Values.shape)

        # Input #
        # Q : (BS, Q_len, n_head, d_K)
        # K : (BS, K_len, n_head, d_K), here, d_K = head dimension
        e = torch.einsum("nqhd,nkhd->nhqk", [Queries,Keys])
        # Output #
        # QK^T = (BS, n_head, Q_len, K_len)
        #print("e, Mask : ", e.shape, Mask.shape)
        if Mask is not None :
            e = e.masked_fill(Mask == 0, float("-1e20"))    # replace b in a, (a, b)
        # The attention score : (BS,n_head, Q_len, K_len), query attention relative to the keys
        # Since the Q = target sentence, K = source sentence.
        alpha = torch.softmax(e/(self.d_model**0.5), dim= 3)
        #print("alpha, value : ", alpha.shape, Values.shape)
        # e : (BS, n_head, Q_len, K_len)
        # V : (BS, K_len, n_head, d_K)
        alpha = torch.einsum("BNQK,BKND->BQND", [alpha, Values]).reshape(
            BS, l_Q, self.n_head*self.d_K
        )
        # (QK^T/sqrt(dim))*V -> (BS, n_head, Q_len, d_K) -> (BS, Q_len, n_head, d_K)
        #print("after einsum : ", alpha.shape)
        out = self.fc_out(alpha) # self.n_head*self.d_K -> self.d_model
        #print("MHA output : " , out.shape)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout = 0.3, forward_expansion = 2):
        super(TransformerBlock, self).__init__()

        self.MHA = MultiHeadAttention(d_model, n_head)
        self.Lnorm1 = nn.LayerNorm(d_model)     # Layernorm : example별 normalization
                                                # Expansion : 일반 convnet의 filter 증가 - 감소와 비슷한 효과!
        self.FFNN   = nn.Sequential(
            nn.Linear(d_model, forward_expansion*d_model),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(forward_expansion*d_model, d_model)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.Lnorm2 = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, Mask):
        attention = self.MHA(Q, K, V, Mask)
        X = self.dropout(self.Lnorm1(attention + Q)) # Since the Q == input from the Emb layer!
        forward = self.FFNN(X)
        out = self.dropout(self.Lnorm2(forward + X))

        return out


class Encoder(nn.Module):
    def __init__(self,
                 source_vocab_size,
                 embed_size,                        # d_model!
                 n_layers,
                 n_heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_len
                 ):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.device     = device
        self.WE         = nn.Embedding(source_vocab_size, embed_size)
        self.PE         = nn.Embedding(max_len, embed_size)

        self.Layers     = nn.ModuleList(
            [
                TransformerBlock(embed_size, n_heads, dropout=dropout, forward_expansion=forward_expansion)
                for i in range(n_layers)
            ]
        )
        self.dropout    = nn.Dropout(dropout)

    def forward(self, X, Mask):
        BS, seq_len = X.shape
        Position = torch.arange(0,seq_len).expand(BS, seq_len).to(self.device) # expand : https://seducinghyeok.tistory.com/9
        out = self.dropout(self.WE(X) + self.PE(Position))
        # 굳이 PE를 sin/cos로 안줘도 알아서 찾음.

        for layer in (self.Layers):
            out = layer(out, out, out, Mask)            # Since we're doing self MHA in Encoder

        return out


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.MHA  = MultiHeadAttention(d_model,n_head)
        self.norm = nn.LayerNorm(d_model)
        self.transformer_block = TransformerBlock(d_model, n_head, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V, source_mask, target_mask):
        '''
        :param X: GT input @ training phase
        :param K: From Encoder features
        :param V: From Encoder features
        :param source_mask: Padding된 것끼리 계산 안하도록
        :param target_mask: LHA mask in Enc-Dec attention
        :return:
        '''
        decoder_attn = self.MHA(X, X, X, target_mask)   # Input : Target, self attention 단계
        Q            = self.dropout(self.norm(decoder_attn + X))
        out          = self.transformer_block(Q, K, V, source_mask)

        return out


class Decoder(nn.Module):
    def __init__(self,
                 target_vocab_size,
                 embed_size,
                 n_layers,
                 n_heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_len):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.WE = nn.Embedding(target_vocab_size, embed_size)
        self.PE = nn.Embedding(max_len, embed_size)
        self.Layers = nn.ModuleList([
            DecoderBlock(embed_size, n_heads, forward_expansion, dropout, device)
            for i in range(n_layers)
        ])
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, enc_out, source_mask, target_mask):
        BS, seq_len = X.shape
        Position = torch.arange(0,seq_len).expand(BS, seq_len).to(self.device)
        out = self.dropout((self.WE(X) + self.PE(Position)))

        for layer in self.Layers:
            out = layer(out, enc_out, enc_out, source_mask, target_mask)    # Q : Decoder // K,V : Encoder

        out = self.fc_out(out)

        return out

class Transformer(nn.Module):
    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 source_pad_idx,
                 target_pad_idx,
                 embed_size = 512,
                 n_layers = 2,
                 forward_expansion = 4,
                 n_heads = 8,
                 dropout = 0.3,
                 device = "cuda",
                 max_len = 100
                 ):
        super(Transformer, self).__init__()
        self.Encoder = Encoder(source_vocab_size, embed_size,
                               n_layers, n_heads, device, forward_expansion,
                               dropout, max_len)
        self.Decoder = Decoder(target_vocab_size, embed_size,
                               n_layers, n_heads, device, forward_expansion,
                               dropout, max_len)
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_source_mask(self, source):
        source_mask = (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2)
        #print("source shape : ", source.shape)
        # (BS, 1, 1, source_length)
        return source_mask.to(self.device)

    def make_target_mask(self, target):
        BS, target_len = target.shape
        #print("target shape : ", target.shape)
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            BS, 1, target_len, target_len)
        return target_mask.to(self.device)

    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)
        enc_feature = self.Encoder(source, source_mask)
        output      = self.Decoder(target, enc_feature, source_mask, target_mask)

        return output


'''
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 1 : sos / 0 : pad / 2 : eos
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)'''