import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stop
from .utils import get_activation_fn, trunc_normal_

############### Encoder ###############
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        # Custom method to return attn outputs. Otherwise same as nn.TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, src, src_mask= None, src_key_padding_mask = None):
        src2,attn = self.self_attn(src, src, src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2) 
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attn
        

class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, nhead = 4,dropout=0.1):
        super().__init__()
        self.transformer_layer = TransformerEncoderLayer(d_model, nhead, d_model*1, dropout=dropout, activation='relu')
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout=dropout, activation='gelu') 

    def forward(self,k,mask=None,padding_mask=None):
        attn = None
        k=k.transpose(0,1)  
        x,attn = self.transformer_layer(k,src_mask=mask,src_key_padding_mask=padding_mask)
        # x = self.transformer_layer(k,src_mask=mask)
        x=x.transpose(0,1)
        return x,attn

############### Decoder ###############
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, mask=None):
        B, N, C = q.shape
        B_k, N_k, C_k = k.shape
        B_v, N_v, C_v = v.shape

        qkv = self.qkv(q).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, N, 3, head_num, head_dim -> 3, B, head_num, N, head_dim
        q = qkv[0]
        qkv = self.qkv(k).reshape(B_k, N_k, 3, self.num_heads, C_k // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, N, 3, head_num, head_dim -> 3, B, head_num, N_K, head_dim
        k = qkv[1]
        qkv = self.qkv(v).reshape(B_v, N_v, 3, self.num_heads, C_v // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, N, 3, head_num, head_dim -> 3, B, head_num, N_V, head_dim
        v = qkv[2]  # v B, head_num, N_V, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn q(B, head_num, N, head_dim) @ k(B, head_num, head_dim, N_k) -> (B, head_num, N, N_k)

        if mask is not None:
            # mask = mask.unsqueeze(1)
            # mask = mask.repeat(1,self.num_heads,1,1)
            msk = torch.tril(torch.ones(B, self.num_heads, N, N_k), diagonal=0).to(attn.device)
            attn = attn.masked_fill(msk == 0, -1e9)
            # attn = attn.masked_fill(mask == True, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # (B, head_num, N, N_k) @ (B, head_num, N_v, head_dim) -> (B, head_num, N, head_dim)->(B, N, emb_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_head = num_heads
        self.norm1 = norm_layer(dim)
        self.slf_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.multihead_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, tgt, enc_output, mask=None):
        # # import pdb
        # # pdb.set_trace()
        # # t0=time()
        # tgt = tgt + self.slf_attn(self.norm1(tgt),self.norm1(tgt),self.norm1(tgt),mask)[0]
        # # t1=time()
        # tgt = tgt + self.multihead_attn(self.norm2(tgt),enc_output,enc_output)[0]
        # # t2=time()
        # tgt = tgt + self.mlp(self.norm3(tgt))
        # return tgt
        tgt = self.norm1(tgt)
        tgt2 = self.slf_attn(tgt, tgt, tgt, mask)[0]
        tgt = tgt + self.drop_path(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt, enc_output, enc_output)[0]
        tgt = tgt + self.drop_path(tgt2)
        tgt = self.norm3(tgt)
        tgt = tgt + self.drop_path(self.mlp(tgt))
        return tgt

