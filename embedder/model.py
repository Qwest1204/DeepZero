import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # -> 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # -> 64 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # -> 128 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),         # -> 256 x 4 x 4
            nn.ReLU(),
        )
        self.enc_out_size = 256 * (img_size // 16) * (img_size // 16)  # для 64: 256*4*4=4096
        self.fc_mu = nn.Linear(self.enc_out_size, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.enc_out_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> 128 x 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # -> 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # -> in_channels x 64 x 64
            nn.Sigmoid()   # для пикселей в диапазоне [0,1]
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)   # flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, self.img_size // 16, self.img_size // 16)  # восстановим форму
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @staticmethod
    def loss_vae(recon_x, x, mu, logvar, beta=1.0):

        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss