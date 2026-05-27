import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CarRacingSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, obs, act, seq_len=8):
        
        self.obs = obs      
        self.actions = act   
        self.seq_len = seq_len

        assert len(self.obs) == len(self.actions) + 1, \
            f"Длина obs ({self.obs[0]}) должна быть на 1 больше длины actions ({len(self.actions)})"
        assert len(self.obs) > self.seq_len, \
            f"Всего наблюдений {self.obs[0]}, нужно seq_len+1 = {seq_len+1}"

        self.num_samples = len(self.obs) - self.seq_len - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        t = idx
        obs_window = self.obs[t : t + self.seq_len]
        act_window = self.actions[t : t + self.seq_len] 
        target_obs = self.obs[t + self.seq_len]
        
        actions = torch.from_numpy(act_window).long()

        return obs_window, actions[:, :3], actions[:, 3:], torch.tensor(target_obs)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.size(1)
        return x + self.pe[:, :S, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model должно делиться на num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, is_causal=False) -> torch.Tensor:

        B, S, _ = query.shape
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)

        return self.out_proj(attn_output)
    
    
class SimpleFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out
    

class PredictorTransformer(nn.Module):
    def __init__(self, z_dim, act_dim, d_model, act_space, n_layer, n_head, max_len):
        super().__init__()
        
        self.act_embedder = nn.Linear(act_space, act_dim)
        self.in_proj = nn.Linear(z_dim+act_dim, d_model)
        self.out_proj_z = nn.Linear(d_model, z_dim)
        self.out_proj_reward = nn.Linear(d_model, 1)
        
        self.pe = SinusoidalPositionalEncoding(d_model, max_len)
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=n_head, dropout=0.1)
        self.ffn = SimpleFFN(d_model, 4*d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    
    def forward(self, z, actions):
        act_emb = self.act_embedder(actions)
        in_x = torch.cat([z, act_emb], dim=-1)
        in_proj_x = self.in_proj(in_x)
        
        x = self.pe(in_proj_x)
        res = x
        
        x = self.norm1(x)

        attn_out = self.mha(x, x, x)
        
        x = res + self.dropout(attn_out)
        
        res = x
        
        x = self.norm2(x)
        ff_out = self.ffn(x)
        
        out_x = res+self.dropout(ff_out)
        
        out_proj_z = self.out_proj_z(out_x)
        reward = self.out_proj_reward(out_x)
        return out_proj_z, reward
        
        