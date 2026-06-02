import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Controller(nn.Module):
    def __init__(self, z_dim, action_dim):
        super().__init__()
        self.nn = nn.Linear(z_dim * 2, action_dim)

    def forward(self, z_in, z_tgt):
        in_nn = torch.cat([z_in, z_tgt], dim=-1)
        hidden = self.nn(in_nn)
        return torch.tanh(hidden)   # действие в [-1,1]
        mean, log_std, value = self.forward(z_in, z_tgt)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action_clamped = action.clamp(-0.999, 0.999)
        z = torch.atanh(action_clamped)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, value