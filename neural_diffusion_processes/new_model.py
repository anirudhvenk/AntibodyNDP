import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.depth = hidden_dim*2 // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim*2)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim*2)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim*2)
        self.final_proj = nn.Linear(hidden_dim*2, hidden_dim*2)
        
        self.scale = self.depth ** -0.5
    
    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # Transpose for attention dot product: (batch_size, num_heads, seq_len, depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v):
        batch_size = q.size(0)
        
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scores = matmul_qk * self.scale
        attn = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attn, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.depth)
        
        return self.final_proj(output)

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(AttentionBlock, self).__init__()
        
        self.t_dense = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        
        self.mh_att = CustomMultiHeadAttention(hidden_dim, num_heads)

    def forward(self, s, t):
        t = self.t_dense(t)[:, None, :]
        y = s + t
        y_att_d = self.mh_att(y, y, y)

        residual, skip = torch.chunk(y_att_d, 2, dim=-1)
        residual = nn.GELU()(residual)
        skip = nn.GELU()(skip)
        
        return (s + residual) / math.sqrt(2.0), skip

class NewAttentionModel(nn.Module):
    def __init__(self, n_layers, hidden_dim, num_heads, input_dim, output_dim):
        super(NewAttentionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.x_dense = nn.Linear(in_features=input_dim+output_dim, out_features=hidden_dim)
        self.att_layers = nn.ModuleList([AttentionBlock(hidden_dim, num_heads) for _ in range(n_layers)])
        self.eps_dense1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.eps_dense2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
    
    def forward(self, x, y, t, mask):
        """
        Computes the additive noise that was added to `y_0` to obtain `y_t`
        based on `x_t` and `y_t` and `t`
        """
        del mask
        
        x = torch.concatenate([x, y], axis=-1)
        x = self.x_dense(x)
        x = nn.GELU()(x)

        t_embedding = timestep_embedding(t, self.hidden_dim)

        skip = None
        for layer in self.att_layers:
            x, skip_connection = layer(x, t_embedding)
            skip = skip_connection if skip is None else skip_connection + skip

        eps = skip / math.sqrt(self.n_layers * 1.0)
        eps = nn.GELU()(self.eps_dense1(eps))
        eps = self.eps_dense2(eps)

        return eps
