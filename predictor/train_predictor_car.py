"""Train the world-model predictor on CarRacing recordings.

Pipeline:
    1. Encode every saved frame with a frozen CarRacing VAE to ``car-z{id}.npy``
       (``(N, 2, *latent_shape)`` pair: ``[0]=mu``, ``[1]=logvar``).
    2. Build a :class:`RecordingDataset` over the latent pairs.
    3. Train ``PredictorTransformer`` with the MDN NLL over the next-latent pair
       plus the Gaussian NLL of a reward head.
    4. Save the best checkpoint via ``save_pretrained`` (config + safetensors).

Example:
    uv run python predictor/train_predictor_car.py \\
        --vae-weights weights/CR/model_033 \\
        --epochs 20 --batch-size 16 --seq-len 32
"""

import argparse
import random
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchinfo import summary
from dataset import RecordingDataset, precompute_z
from embedder import VAE
from predictor import PredictorConfig, PredictorTransformer
from tqdm import tqdm

# Reward head weight (relative to the MDN NLL).
REWARD_LAMBDA = 0.4
# Entropy bonus to prevent MDN mode collapse (encourages diverse components).
ENTROPY_LAMBDA = 0.005
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Train the CarRacing latent dynamics predictor")
    p.add_argument("--data-dir", default="try", help="root data directory")
    p.add_argument("--vae-weights", default="weights/CR/model_033", help="pretrained CR VAE dir")
    p.add_argument("--save-dir", default="weights/predictor_car", help="checkpoint output dir")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128, help="window length (= config.max_len)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-gaussians", type=int, default=4)
    p.add_argument("--act-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-reward", action="store_true", help="disable the reward head")
    p.add_argument("--val-frac", type=float, default=0.05, help="fraction of windows for validation")
    p.add_argument("--device", default=None,
                   help="torch device (default: mps > cuda > cpu)")
    return p.parse_args()


def load_vae(args, device):
    print(f"[vae] loading {args.vae_weights}")
    vae = VAE.from_pretrained(args.vae_weights)
    vae.eval().to(device)
    return vae


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}")

    vae = load_vae(args, device)

    print("[precompute] encoding CarRacing sessions to latent pairs ...")
    written = precompute_z(args.data_dir, vae=vae, device=device, game="car", batch=32)
    if not written:
        raise SystemExit("Нет car-сессий для предиктора (сначала запишите: uv run python -m games.record car)")

    # Latent shape follows the z-file (spatial for CR models, flat for legacy).
    z0 = np.load(written[0])
    latent_shape = tuple(z0.shape[2:]) if z0.ndim == 5 else (int(z0.shape[2]),)
    print(f"[latent] latent_shape={latent_shape}, z_dim={int(np.prod(latent_shape))}")

    print("[dataset] building RecordingDataset ...")
    full = RecordingDataset(
        args.data_dir, game="car", seq_len=args.seq_len, mode="predictor",
        one_hot_act=False,
    )
    if len(full) == 0:
        raise SystemExit("Dataset пуст (нет окон после done-фильтра)")

    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    train_sub, val_sub = random_split(full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(SEED))
    train_dl = DataLoader(Subset(full, train_sub.indices), batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(Subset(full, val_sub.indices), batch_size=args.batch_size, shuffle=False)

    cfg = PredictorConfig(
        latent_shape=latent_shape,
        act_space=3,          # Car: (steer, gas, brake), continuous
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_gaussians=args.n_gaussians,
        max_len=args.seq_len,
        act_dim=args.act_dim,
        dropout=args.dropout,
        predict_reward=not args.no_reward,
    )
    model = PredictorTransformer(cfg).to(device)
    summary(model)
    print(f"[model] params={model.num_parameters() / 1e6:.2f}M  cfg={cfg}")
    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_nll = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_mdn, tot_rw, tot_all, tot_ent = 0.0, 0.0, 0.0, 0.0
        nb = 0
        for mu, logvar, act, tgt_mu, tgt_logvar, rew in tqdm(train_dl):
            mu = mu.to(device); logvar = logvar.to(device); act = act.to(device)
            tgt_mu = tgt_mu.to(device); tgt_logvar = tgt_logvar.to(device)

            pi, pred_mu, pred_logvar, pred_rew = model(mu, logvar, act, mode="all")
            mdn = model.mdn_loss(pi, pred_mu, pred_logvar, tgt_mu, tgt_logvar)

            pi_entropy = -(pi * (pi + 1e-10).log()).sum(-1).mean()

            if pred_rew is not None and rew is not None:
                rw_loss = model.reward_loss(pred_rew[..., 0], pred_rew[..., 1], rew.to(device))
            else:
                rw_loss = torch.zeros(())
            loss = mdn + REWARD_LAMBDA * rw_loss - ENTROPY_LAMBDA * pi_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_mdn += mdn.item()
            tot_rw += (rw_loss.item() if rw_loss.numel() else 0.0)
            tot_all += loss.item()
            tot_ent += pi_entropy.item()
            nb += 1

        val_nll, val_ent = evaluate(model, val_dl, device)
        print(f"epoch {epoch:3d} | mdn {tot_mdn / nb:9.3f} | rew {tot_rw / nb:9.4f} | "
              f"ent {tot_ent / nb:.3f} | loss {tot_all / nb:9.3f} | "
              f"val_mdn {val_nll:9.3f} | val_ent {val_ent:.3f}")

        if val_nll < best_nll:
            best_nll = val_nll
            os.makedirs(args.save_dir, exist_ok=True)
            model.save_pretrained(args.save_dir)
            print(f"  -> saved checkpoint (val_mdn {val_nll:.3f}) to {args.save_dir}")

    print(f"DONE best val_mdn = {best_nll:.3f}  saved to {args.save_dir}")


@torch.no_grad()
def evaluate(model, val_dl, device):
    model.eval()
    nlls, ents = [], []
    for mu, logvar, act, tgt_mu, tgt_logvar, _ in val_dl:
        pi, pred_mu, pred_logvar, _ = model(mu.to(device), logvar.to(device),
                                            act.to(device), mode="all")
        nlls.append(mdn_loss(pi, pred_mu, pred_logvar, tgt_mu.to(device),
                             tgt_logvar.to(device)).item() * mu.size(0))
        pi_e = -(pi * (pi + 1e-10).log()).sum(-1).mean()
        ents.append(pi_e.item() * mu.size(0))
    model.train()
    n = max(1, len(val_dl.dataset))
    return sum(nlls) / n, sum(ents) / n


def mdn_loss(pi, mu, logvar, tgt_mu, tgt_logvar):
    """Convenience wrapper around the static loss (flat/spatial agnostic)."""
    from predictor.model import PredictorTransformer
    return PredictorTransformer.mdn_loss(pi, mu, logvar, tgt_mu, tgt_logvar)


if __name__ == "__main__":
    main()