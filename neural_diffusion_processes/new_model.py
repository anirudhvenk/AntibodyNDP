import math
import torch
import torch.nn as nn
from einops import rearrange, reduce

def timestep_embedding(t, embedding_dim, max_positions=10_000):
    """Sinusoidal embedding"""
    half_dim = embedding_dim // 2
    emb = torch.log(torch.as_tensor([max_positions])) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.concatenate([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (t.shape[0], embedding_dim)
    return emb

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def forward(self, s, t):
        t = nn.Linear(in_features=t.shape[-1], out_features=self.hidden_dim)(t)[:, None, :]
        y = s + t
        
        q = nn.Linear(in_features=y.shape[-1], out_features=self.hidden_dim*2)(y) 
        k = nn.Linear(in_features=y.shape[-1], out_features=self.hidden_dim*2)(y)
        v = nn.Linear(in_features=y.shape[-1], out_features=self.hidden_dim*2)(y)
        
        y_att_d = nn.MultiheadAttention(self.hidden_dim*2, self.num_heads, batch_first=True)(q, v, k)
        y = y_att_d

        residual, skip = torch.chunk(y[0], 2, dim=-1)
        residual = nn.GELU()(residual)
        skip = nn.GELU()(skip)
        
        return (s + residual) / math.sqrt(2.0), skip

class NewAttentionModel(nn.Module):
    def __init__(self, n_layers, hidden_dim, num_heads, output_dim):
        super(NewAttentionModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
    
    def forward(self, x, y, t, mask):
        """
        Computes the additive noise that was added to `y_0` to obtain `y_t`
        based on `x_t` and `y_t` and `t`
        """
        del mask
        print("Shape: ", x.shape)
        print("Shape: ", y.shape)
        
        x = torch.concatenate([x, y], axis=-1)
        x = nn.Linear(in_features=x.shape[-1], out_features=self.hidden_dim)(x)
        x = nn.GELU()(x)

        t_embedding = timestep_embedding(t, self.hidden_dim)

        skip = None
        for _ in range(self.n_layers):
            layer = AttentionBlock(self.hidden_dim, self.num_heads)
            x, skip_connection = layer(x, t_embedding)
            skip = skip_connection if skip is None else skip_connection + skip

        eps = skip / math.sqrt(self.n_layers * 1.0)
        eps = nn.GELU()(nn.Linear(in_features=eps.shape[-1], out_features=self.hidden_dim)(eps))
        eps = nn.Linear(in_features=eps.shape[-1] ,out_features=self.output_dim)(eps)

        return eps
