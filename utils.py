import torch
from typing import Tuple

def batched_vae_encode(model: torch.nn.Module, 
                       tensor: torch.Tensor, 
                       batch_size: int = 64) -> torch.Tensor:
    n_samples = tensor.size(0)
    device = tensor.device
    model.eval()
    outputs = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = tensor[i:i + batch_size].to(device)
            mean, log_var = model.encode(batch)
            latent = model.reparameterize(mean, log_var)
            latent = latent.squeeze()
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            outputs.append(latent)

    result = torch.cat(outputs, dim=0)
    return result