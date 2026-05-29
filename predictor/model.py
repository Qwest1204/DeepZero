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
            f"Длина obs ({len(self.obs)}) должна быть на 1 больше длины actions ({len(self.actions)})"
        assert len(self.obs) > self.seq_len, \
            f"Всего наблюдений {len(self.obs)}, нужно seq_len+1 = {seq_len+1}"

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


# ---------- Новый класс: один блок Transformer ----------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SimpleFFN(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.mha(x, x, x)          # self-attention
        x = self.dropout1(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x
        return x


# ---------- Основная модель с несколькими слоями ----------
class PredictorTransformer(nn.Module):
    def __init__(self, z_dim, act_dim, d_model, act_space, n_layer, n_head, max_len):
        super().__init__()
        
        self.act_embedder = nn.Linear(act_space, act_dim)
        self.in_proj = nn.Linear(z_dim + act_dim, d_model)
        
        # Позиционное кодирование (один раз для всей последовательности)
        self.pe = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Стек из n_layer блоков Transformer
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, 4 * d_model, dropout=0.1)
            for _ in range(n_layer)
        ])
        
        # Выходные проекции
        self.mu = nn.Linear(d_model, z_dim)
        self.logvar = nn.Linear(d_model, z_dim)
    
    def forward(self, z, actions):
        # z: (B, S, z_dim), actions: (B, S, act_space)
        act_emb = self.act_embedder(actions)                     # (B, S, act_dim)
        in_x = torch.cat([z, act_emb], dim=-1)                   # (B, S, z_dim+act_dim)
        x = self.in_proj(in_x)                                   # (B, S, d_model)
        
        x = self.pe(x)                                            # добавляем позиционную информацию
        
        # Последовательно пропускаем через все слои
        for layer in self.layers:
            x = layer(x)
        
        # Финальные линейные проекции на предсказание z и награды
        mu = self.mu(x)                               # (B, S, z_dim)
        logvar = self.logvar(x)                         # (B, S, 1)
        return mu, logvar