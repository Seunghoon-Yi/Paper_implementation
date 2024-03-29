import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Customized_S3DG import S3DG
from C3D_model import C3D

'''
class C3D(nn.Module):
    def __init__(self, d_model, device, dropout, length_limit):
        super(C3D, self).__init__()
        
        Input  : [BS, T, C, H, W] = [8, 240, 224, 192, 3]
        Output : [BS, C, T, H, W] = [BS, 128, 75,  7,   6] -> [BS, 75, 1024] 
        
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # BS, 8, 300, 240, 200
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # BS, 8, 300, 120, 100
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # BS, 16, 300, 120, 100
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # BS, 16, 300, 60, 50
        self.conv3a = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # BS, 32, 300, 58, 48
        self.conv3b = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # BS, 64, 300, 56, 48
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # BS, 64, 150, 28, 24

        self.conv4a = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # BS, 128, 150, 28, 24
        self.conv4b = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # BS, 256, 150, 28, 24
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # BS, 256, 150, 14, 12
        self.conv5a = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # BS, 256, 150, 14, 12
        self.conv5b = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # BS, 128, 150, 14, 12
        self.pool5 = nn.AdaptiveMaxPool3d(output_size=(60, 7, 6))
        # BS, 128, 75, 7, 6
        # 0925 : (BS, 64, 60, 7, 6)

        self.fc6     = nn.Linear(64*7*6, 1024)
        self.fc7     = nn.Linear(1024, d_model)
        self.dropout1= nn.Dropout(p=0.5)
        self.dropout2= nn.Dropout(p=0.5)
        self.lrelu   = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        #self.device  = device

    def forward(self, X):
        X = X.permute(0,2,1,3,4)

        X = self.lrelu(self.conv1(X))
        X = self.pool1(X)

        X = self.lrelu(self.conv2(X))
        X = self.pool2(X)

        X = self.lrelu(self.conv3a(X))
        X = self.lrelu(self.conv3b(X))
        X = self.pool3(X)

        X = self.lrelu(self.conv4a(X))
        X = self.lrelu(self.conv4b(X))
        X = self.pool4(X)
        #print(X.shape)

        X = self.lrelu(self.conv5a(X))
        X = self.lrelu(self.conv5b(X))
        X = self.pool5(X)
        #print(X.shape)

        X = X.permute(0,2,1,3,4)
        X = X.reshape(-1, 60, 64*7*6) # BS, T, rest of them
        X = self.dropout1(X)
        out = self.lrelu(self.fc6(X))
        out = self.dropout2(out)
        out = self.lrelu(self.fc7(out))

        return out
'''


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, device):
        super(MultiHeadAttention, self).__init__()

        self.d_K = d_model//n_head
        self.n_head = n_head
        self.d_model = d_model

        assert (self.n_head*self.d_K == self.d_model), "Embed size needs to be divisible by heads"
        self.WQ = nn.Linear(self.d_model, self.d_model, bias=True)
        self.WK = nn.Linear(self.d_model, self.d_model, bias=True)
        self.WV = nn.Linear(self.d_model, self.d_model, bias=True)
        # The W0
        self.fc_out  = nn.Linear(self.d_K*self.n_head, d_model)
        self.dropout = nn.Dropout(p = dropout)
        self.scale   = torch.sqrt(torch.FloatTensor([self.d_K])).to(device)

    def forward(self, Q_ipt, K_ipt, V_ipt, Mask = None):
        BS = Q_ipt.shape[0]
        l_Q, l_K, l_V = Q_ipt.shape[1], K_ipt.shape[1], V_ipt.shape[1]
        Q = self.WQ(Q_ipt)
        K = self.WK(K_ipt)
        V = self.WV(V_ipt) # (BS, seq_len, d_model)

        # Split inputs to n_heads
        Queries = Q.reshape(BS, l_Q, self.n_head, self.d_K).permute(0, 2, 1, 3)
        Keys    = K.reshape(BS, l_K, self.n_head, self.d_K).permute(0, 2, 1, 3)
        Values  = V.reshape(BS, l_V, self.n_head, self.d_K).permute(0, 2, 1, 3)
        #dim = [BS, n_head, seq_len, d_k]

        # Input #
        e = torch.matmul(Queries, Keys.permute(0,1,3,2)) / self.scale

        if Mask is not None :
            e = e.masked_fill(Mask == 0, float("-1e10"))    # replace b in a, (a, b)
        alpha = torch.softmax(e, dim= -1)
        #dim = [BS, n_head, l_Q, l_K]
        out = torch.matmul(self.dropout(alpha), Values)
        #dim = [BS, n_head, l_Q, d_K]

        out = out.permute(0,2,1,3).contiguous()
        #dim = [BS, l_Q, n_head, d_K]
        out = out.view(BS, -1, self.d_model)
        out = self.fc_out(out)
        #dim = [BS, l_Q, d_model]

        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout, device, forward_expansion = 2):
        super(TransformerBlock, self).__init__()

        self.MHA = MultiHeadAttention(d_model, n_head, dropout=dropout, device = device)
        self.Lnorm1 = nn.LayerNorm(d_model)     # Layernorm : example별 normalization
                                                # Expansion : 일반 convnet의 filter 증가 - 감소와 비슷한 효과!
        self.FFNN   = nn.Sequential(
            nn.Linear(d_model, forward_expansion*d_model),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p = dropout),
            nn.Linear(forward_expansion*d_model, d_model)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.Lnorm2 = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, Mask):
        attention = self.MHA(Q, K, V, Mask)
        X = self.Lnorm1(self.dropout(attention) + Q) # Since the Q == input from the Emb layer!
        forward = self.FFNN(X)
        out = self.Lnorm2(self.dropout(forward) + X)

        return out

class Encoder(nn.Module):
    def __init__(self,
                 gloss_vocab_size,
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
        #self.WE         = nn.Linear(1024, embed_size)
        self.PE         = nn.Embedding(max_len, embed_size)
        #self.pool       = nn.AdaptiveMaxPool2d((55, embed_size))
        self.fc_out     = nn.Linear(embed_size, gloss_vocab_size) # For gloss
        #self.softmax    = nn.Softmax(dim = -1)

        self.Layers     = nn.ModuleList(
            [
                TransformerBlock(embed_size, n_heads, dropout=dropout,
                                 device=device, forward_expansion=forward_expansion)
                for i in range(n_layers)
            ]
        )
        self.dropout    = nn.Dropout(dropout)
        self.scale      = torch.sqrt(torch.FloatTensor([embed_size])).to(device)

    def forward(self, X, Mask):
        BS, seq_len, emb_dim = X.shape

        Position = torch.arange(0,seq_len).expand(BS, seq_len).to(self.device) # expand : https://seducinghyeok.tistory.com/9

        out = self.dropout(X * self.scale + self.PE(Position))
        #print("after encoder enc : ", out.shape, Mask.shape)

        for layer in (self.Layers):
            out = layer(out, out, out, Mask)            # Since we're doing self MHA in Encoder

        # gloss output
        #pool_out      = self.pool(out)
        predict_gloss = self.fc_out(self.dropout(out))
        #predict_gloss = self.softmax(predict_gloss)

        return out, predict_gloss



class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.MHA  = MultiHeadAttention(d_model,n_head, dropout=dropout,device=device)
        self.norm = nn.LayerNorm(d_model)
        self.transformer_block = TransformerBlock(d_model, n_head,dropout=dropout,
                                device=device, forward_expansion=forward_expansion)
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
        #print(X.shape, target_mask.shape)
        decoder_attn = self.MHA(X, X, X, target_mask)   # Input : Target, self attention 단계
        Q            = self.norm(self.dropout(decoder_attn) + X)
        out          = self.transformer_block(Q, K, V, source_mask)

        return out

class Decoder(nn.Module):
    def __init__(self,
                 target_vocab_size,
                 gloss_vocab_size,
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
        #self.gloss_decoder =\
        #DecoderBlock(embed_size, n_heads, forward_expansion, dropout, device)
        self.fc_out  = nn.Linear(embed_size, target_vocab_size)
        #self.fc_gloss = nn.Linear(embed_size, gloss_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale   = torch.sqrt(torch.FloatTensor([embed_size])).to(device)
        #self.softmax = nn.Softmax(dim = -1)

    def forward(self, X, enc_out, source_mask, target_mask):
        BS, seq_len = X.shape

        Position = torch.arange(0,seq_len).expand(BS, seq_len).to(self.device)
        out = self.dropout((self.WE(X) * self.scale + self.PE(Position)))

        #gloss_attn = self.gloss_decoder(out, enc_out, enc_out, source_mask, target_mask)
        #predict_gloss = self.fc_gloss(gloss_attn)

        #print("after decoder enc : ", out.shape)
        for layer in self.Layers:
            out = layer(out, enc_out, enc_out, source_mask, target_mask)    # Q : Decoder // K,V : Encoder

        out = self.fc_out(out)
        #out = self.softmax(out)

        return out



class SLT_Transformer(nn.Module):
    def __init__(self,
                 gloss_vocab_size,
                 target_vocab_size,
                 source_pad_idx,
                 target_pad_idx,
                 embed_size = 512,
                 n_layers = 2,
                 forward_expansion = 4,
                 n_heads = 8,
                 dropout = 0.25,
                 device = "cuda",
                 max_len_enc =64,
                 max_len_dec =64
                 ):
        super(SLT_Transformer, self).__init__()

        self.C3D     = C3D(embed_size)
        #self.S3D     = S3DG(embed_size, device, dropout = dropout)
        self.Encoder = Encoder(gloss_vocab_size, embed_size,
                               n_layers, n_heads, device, forward_expansion,
                               dropout, max_len_enc)
        self.Decoder = Decoder(target_vocab_size, gloss_vocab_size, embed_size,
                               n_layers, n_heads, device, forward_expansion,
                               dropout, max_len_dec)
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_source_mask(self, source):
        source_mask = (source[:,:,0] != self.source_pad_idx).unsqueeze(1).unsqueeze(2)
        #print("source shape : ", source.shape)
        #print("mask shape : ", source_mask.shape)
        # (BS, 1, 1, source_length)
        return source_mask.to(self.device)

    def make_target_mask(self, target):
        BS, target_len = target.shape
        #print("target shape : ", target.shape)
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            BS, 1, target_len, target_len)
        return target_mask.to(self.device)

    def forward(self, source, target):

        #print("source : ", source.dtype)
        C3D_feature = self.C3D(source)
        #print(source.shape)
        #S3D_feature  = self.S3D(source)
        #print("after c3d : ", C3D_feature.shape)

        source_mask = self.make_source_mask(C3D_feature)
        target_mask = self.make_target_mask(target)

        # Outputs : glosses and translations
        enc_feature, predict_gloss\
                            = self.Encoder(C3D_feature, source_mask)
        predict_translation = self.Decoder(target, enc_feature, source_mask, target_mask)

        return predict_translation, predict_gloss

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

if __name__ == "__main__":
    inputs = torch.rand(2, 240, 3, 224, 192)
    net = C3D(512, "cuda", 0.5, 300)

    outputs = net.forward(inputs)
    print(outputs.shape)