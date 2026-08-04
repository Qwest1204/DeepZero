import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _resize_to_vae(x: torch.Tensor, vae) -> torch.Tensor:
    """Resize batched ``(B, 3, H, W)`` frames to the VAE ``img_size``."""
    img_size = getattr(vae, "img_size", None)
    if img_size is None:
        return x
    if tuple(x.shape[-2:]) != (img_size, img_size):
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x


def _glob_both(data_dir, pattern):
    """Glob in data_dir and its CarRacing/, Doom/ and MW/ subdirectories."""
    return (
        glob.glob(os.path.join(data_dir, pattern))
        + glob.glob(os.path.join(data_dir, "CarRacing", pattern))
        + glob.glob(os.path.join(data_dir, "Doom", pattern))
        + glob.glob(os.path.join(data_dir, "MW", pattern))
    )


def _find_sessions(data_dir, game):
    obs_act_pairs = []

    if game in ("car", None):
        act_files = _glob_both(data_dir, "car-act*.npy")
        for act_path in act_files:
            m = re.search(r"car-act(\d+)\.npy", os.path.basename(act_path))
            if not m:
                continue
            session_id = m.group(1)
            obs_path = os.path.join(os.path.dirname(act_path), f"car-obs{session_id}.npy")
            if os.path.exists(obs_path):
                obs_act_pairs.append((obs_path, act_path))

    if game in ("doom", None):
        act_files = _glob_both(data_dir, "doom-act*.npy")
        for act_path in act_files:
            m = re.search(r"doom-act(\d+)\.npy", os.path.basename(act_path))
            if not m:
                continue
            session_id = m.group(1)
            obs_path = os.path.join(os.path.dirname(act_path), f"doom-obs{session_id}.npy")
            if os.path.exists(obs_path):
                obs_act_pairs.append((obs_path, act_path))

    if game in ("metaworld", "mw", None):
        act_files = _glob_both(data_dir, "metaworld-act*.npy")
        for act_path in act_files:
            m = re.search(r"metaworld-act(\d+)-(.*)\.npy", os.path.basename(act_path))
            if not m:
                continue
            session_id = m.group(1)
            task = m.group(2)
            obs_path = os.path.join(os.path.dirname(act_path), f"metaworld-obs{session_id}-{task}.npy")
            if os.path.exists(obs_path):
                obs_act_pairs.append((obs_path, act_path))
    if not obs_act_pairs:
        available = _glob_both(data_dir, "*.npy")
        names = [os.path.basename(p) for p in available]
        raise FileNotFoundError(
            f"Не найдено пар obs+act в '{data_dir}'.\n"
            f"Искал: car-act*.npy + car-obs*.npy, "
            f"doom-act*.npy + doom-obs*.npy "
            f"или metaworld-act*.npy + metaworld-obs*.npy.\n"
            f"Найденные .npy файлы: {names if names else '(нет .npy)'}"
        )

    return obs_act_pairs


def interp_even_latents(z_even: np.ndarray) -> np.ndarray:
    """Expand subsampled latents (frames 0, 2, 4, ...) to a full sequence.

    Missing odd frames are approximated by linear interpolation:
    ``Z_i = 0.5 * (Z_{i-1} + Z_{i+1})``.

    Args:
        z_even: Latents of saved frames ``(M, z_dim)`` (or ``(M,)``).

    Returns:
        Full latent sequence ``(2M - 1, z_dim)``; for ``M <= 1`` returns a copy.
    """
    z_even = np.asarray(z_even)
    M = len(z_even)
    if M <= 1:
        return z_even.copy()
    if z_even.ndim == 1:
        z_even = z_even[:, None]
        flat = True
    else:
        flat = False
    full = np.empty((2 * M - 1, z_even.shape[-1]), dtype=z_even.dtype)
    full[0::2] = z_even
    full[1::2] = 0.5 * (z_even[:-1] + z_even[1:])
    return full[:, 0] if flat else full


def _interp_even_any(z_even: np.ndarray) -> np.ndarray:
    """Linearly interpolate odd positions of an ``(M, *shape)`` latent array."""
    z_even = np.asarray(z_even)
    M = len(z_even)
    if M <= 1:
        return z_even.copy()
    full = np.empty((2 * M - 1, *z_even.shape[1:]), dtype=z_even.dtype)
    full[0::2] = z_even
    full[1::2] = 0.5 * (z_even[:-1] + z_even[1:])
    return full


def _z_path_for(obs_path: str) -> str:
    """Path of the precomputed latent file for a given obs file (same prefix)."""
    return re.sub(r"-obs", "-z", obs_path)


def precompute_z(data_dir="try", vae=None, device="cpu", camera_idx=1,
                 dtype=np.float16, game=None, batch=32):
    """Encode sessions with a frozen VAE and save latent z-files.

    ``game="car"`` / ``game="doom"``: every saved frame is encoded with the
    VAE mean and written as ``car-z{id}.npy`` / ``doom-z{id}.npy``, in the
    exact latent shape the VAE emits (spatial ``(N, C, H, W)`` for the square
    doom model, flat ``(N, z_dim)`` for the flat-latent car models).

    ``game="mw"``: keeps the existing interpolating path (obs frames are saved
    on even env steps; the odd latents are linearly interpolated).

    Args:
        data_dir: Root data directory (searches ``CarRacing/``, ``Doom/``,
            ``MW/`` subdirectories).
        vae: Frozen VAE instance (``encode(x) -> (mu, logvar)``).
        device: Torch device for encoding.
        camera: Camera index inside the saved (N, C, H, W, 3) MW obs axis.
        dtype: Numpy dtype for the saved latents.
        game: One of ``"car"``, ``"doom"``, ``"mw"`` or ``None``
            (None runs all three).
        batch: Encoding batch size.

    Returns:
        List of written ``*-z*.npy`` paths.
    """
    if vae is None:
        raise ValueError("precompute_z требует vae (from_pretrained VAE)")
    vae = vae.to(device)
    vae.eval()

    if game is None or game in ("car", "doom", "mw"):
        games = ("car", "doom", "mw") if game is None else (game,)
    else:
        raise ValueError(f"Неизвестная игра для precompute_z: {game!r}")

    written = []

    if "mw" in games:
        written += _precompute_mw(data_dir, vae, device, camera_idx, dtype, batch)
    if "car" in games:
        written += _precompute_frames(data_dir, "car", vae, device, dtype, batch)
    if "doom" in games:
        written += _precompute_frames(data_dir, "doom", vae, device, dtype, batch)

    if not written:
        print(f"precompute_z: не найдено сессий для игры {games}")
    return written


def _precompute_frames(data_dir, game, vae, device, dtype, batch):
    """Encode every saved frame of ``game`` sessions to ``{game}-z{id}.npy``.

    Each z-file stores the full posterior pair as ``(N, 2, *latent_shape)``:
    channel ``0`` is the VAE mean and channel ``1`` the log-variance.
    """
    written = []
    act_files = _glob_both(data_dir, f"{game}-act*.npy")
    for act_path in act_files:
        m = re.search(rf"{game}-act(\d+)\.npy", os.path.basename(act_path))
        if not m:
            continue
        session_id = m.group(1)
        obs_path = os.path.join(
            os.path.dirname(act_path), f"{game}-obs{session_id}.npy"
        )
        z_path = _z_path_for(obs_path)
        if not os.path.exists(obs_path):
            continue
        obs_arr = np.load(obs_path)
        if obs_arr.ndim == 5:
            obs_arr = obs_arr[:, 1]
        x = torch.from_numpy(obs_arr).float().to(device) / 255.0
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        x = _resize_to_vae(x, vae)
        mu_parts, lv_parts = [], []
        for i in range(0, len(x), batch):
            with torch.no_grad():
                mu, logvar = vae.encode(x[i : i + batch])
            mu_parts.append(mu.detach().cpu())
            lv_parts.append(logvar.detach().cpu())
        mu = torch.cat(mu_parts, dim=0)
        logvar = torch.cat(lv_parts, dim=0)
        z = torch.stack([mu, logvar], dim=1).numpy().astype(dtype)  # (N, 2, *shape)
        np.save(z_path, z)
        written.append(z_path)
        print(f"precompute_z: {os.path.basename(obs_path)} "
              f"{obs_arr.shape} -> {z.shape} ({z.nbytes / 1e6:.1f} MB)")
    return written


def _precompute_mw(data_dir, vae, device, camera, dtype, batch):
    """Encode subsampled MetaWorld obs and interpolate the odd latents.

    Saves ``metaworld-z{idx}-{task}.npy`` (``(N, 2, *latent_shape)`` pairs)
    next to each obs file.
    """
    written = []
    act_files = _glob_both(data_dir, "metaworld-act*.npy")
    for act_path in act_files:
        m = re.search(r"metaworld-act(\d+)-(.*)\.npy", os.path.basename(act_path))
        if not m:
            continue
        session_id, task = m.group(1), m.group(2)
        obs_path = os.path.join(
            os.path.dirname(act_path), f"metaworld-obs{session_id}-{task}.npy"
        )
        z_path = _z_path_for(obs_path)
        if not os.path.exists(obs_path):
            continue
        obs_arr = np.load(obs_path)
        if obs_arr.ndim == 5:
            obs_arr = obs_arr[:, camera]
        x = torch.from_numpy(obs_arr).float().to(device) / 255.0
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        x = _resize_to_vae(x, vae)
        mu_parts, lv_parts = [], []
        for i in range(0, len(x), batch):
            with torch.no_grad():
                mu, logvar = vae.encode(x[i : i + batch])
            mu_parts.append(mu.detach().cpu())
            lv_parts.append(logvar.detach().cpu())
        mu = torch.cat(mu_parts, dim=0)
        logvar = torch.cat(lv_parts, dim=0)
        pair_even = torch.stack([mu, logvar], dim=1).numpy().astype(dtype)
        pair_full = _interp_even_any(pair_even)
        np.save(z_path, pair_full)
        written.append(z_path)
        print(f"precompute_z: {os.path.basename(obs_path)} "
              f"{obs_arr.shape} -> {pair_full.shape} ({pair_full.nbytes / 1e6:.1f} MB)")
    return written


class RecordingDataset(Dataset):
    def __init__(self, data_dir="try", game=None, seq_len=8, mode="predictor",
                 camera_idx=1, one_hot_act=False):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.mode = mode
        self.camera_idx = camera_idx
        self.one_hot_act = one_hot_act

        pairs = _find_sessions(data_dir, game)

        self.sessions = []
        for obs_path, act_path in pairs:
            act_arr = np.load(act_path)

            # Prefer precomputed latent files (mode="predictor"): same prefix
            # with "obs" replaced by "z" (car-z, doom-z, metaworld-z).
            z_path = None
            if "obs" in os.path.basename(obs_path) and self.mode == "predictor":
                candidate = _z_path_for(obs_path)
                if os.path.exists(candidate):
                    z_path = candidate

            if z_path is not None:
                obs_arr = np.load(z_path)
                fmt = "z"
            else:
                obs_arr = np.load(obs_path)
                # MetaWorld multi-camera obs: (N, C, H, W, 3) -> select one camera
                if obs_arr.ndim == 5:
                    obs_arr = obs_arr[:, self.camera_idx]

            n_obs = len(obs_arr)
            n_act = len(act_arr)

            if z_path is not None:
                if n_obs not in (n_act - 1, n_act, n_act + 1):
                    raise ValueError(
                        f"{os.path.basename(obs_path)}: precomputed z len={n_obs}, "
                        f"len(act)={n_act}. Ожидалось n_obs в (n_act-1, n_act, n_act+1)."
                    )
            elif n_obs == n_act + 1:
                fmt = "car"
            elif n_obs == n_act or (n_obs == (n_act + 1) // 2 and "metaworld" in os.path.basename(obs_path)):
                fmt = "doom" if "metaworld" not in os.path.basename(obs_path) else "metaworld"
            else:
                raise ValueError(
                    f"{os.path.basename(obs_path)}: len(obs)={n_obs}, "
                    f"len(act)={n_act}. Ожидалось n_obs=n_act+1 (CarRacing), "
                    f"n_obs=n_act (Doom/MetaWorld) или "
                    f"n_obs=(n_act+1)//2 (MetaWorld subsampled)."
                )

            subsampled = fmt == "metaworld" and n_obs == (n_act + 1) // 2

            done_arr = None
            reward_arr = None
            if "metaworld" in os.path.basename(obs_path):
                m = re.search(r"metaworld-obs(\d+)-(.*)\.npy", os.path.basename(obs_path))
                if m:
                    done_path = os.path.join(
                        os.path.dirname(obs_path),
                        f"metaworld-success{m.group(1)}-{m.group(2)}.npy",
                    )
                    if os.path.exists(done_path):
                        done_arr = np.load(done_path).astype(bool)
                    reward_path = os.path.join(
                        os.path.dirname(obs_path),
                        f"metaworld-reward{m.group(1)}-{m.group(2)}.npy",
                    )
                    if os.path.exists(reward_path):
                        reward_arr = np.load(reward_path).astype(np.float32)
            elif "car" in os.path.basename(obs_path):
                m = re.search(r"car-obs(\d+)\.npy", os.path.basename(obs_path))
                if m:
                    reward_path = os.path.join(
                        os.path.dirname(obs_path),
                        f"car-reward{m.group(1)}.npy",
                    )
                    if os.path.exists(reward_path):
                        reward_arr = np.load(reward_path).astype(np.float32)

            self.sessions.append((obs_arr, act_arr, fmt, done_arr, subsampled, reward_arr))

        self._build_index()

    def _build_index(self):
        cumlens = []
        valid_sessions = []

        for obs, act, fmt, done, subsampled, reward in self.sessions:
            if self.mode == "vae":
                allowed = np.arange(len(obs))
            else:
                n_samples = len(obs) - self.seq_len - 1
                allowed = np.arange(n_samples)
                # drop windows crossing an episode boundary (success/done)
                if done is not None and self.seq_len > 0:
                    if fmt == "metaworld" and subsampled:
                        # done is per env step; obs frames are steps 0, 2, 4, ...
                        # transition frame j -> j+1 covers steps 2j and 2j+1
                        step = np.zeros(2 * len(obs), dtype=bool)
                        step[: len(done)] = done[: len(done)]
                        trans_bad = step.reshape(len(obs), 2).any(axis=1)
                        bad = np.convolve(
                            trans_bad.astype(int),
                            np.ones(self.seq_len, dtype=int),
                            mode="valid",
                        ) > 0
                    else:
                        d = done[: len(obs)].astype(int)
                        kernel = self.seq_len + 1 if fmt == "car" else self.seq_len
                        bad = np.convolve(d, np.ones(kernel, dtype=int), mode="valid") > 0
                    allowed = allowed[~bad[: len(allowed)]]

            if len(allowed) > 0:
                prev = cumlens[-1] if cumlens else 0
                cumlens.append(prev + len(allowed))
                valid_sessions.append((obs, act, fmt, done, subsampled, reward, allowed))

        self.sessions = valid_sessions
        self.cumlen = np.array(cumlens, dtype=int)
        self.total = int(self.cumlen[-1]) if self.cumlen.size > 0 else 0

    def __len__(self):
        return self.total

    def _locate(self, idx):
        session_idx = int(np.searchsorted(self.cumlen, idx + 1))
        offset = int(self.cumlen[session_idx - 1]) if session_idx > 0 else 0
        frame_idx = idx - offset
        return session_idx, frame_idx

    def __getitem__(self, idx):
        session_idx, frame_t = self._locate(idx)
        obs_arr, act_arr, fmt, done, subsampled, reward, allowed = self.sessions[session_idx]
        frame_t = int(allowed[frame_t])

        if self.mode == "vae":
            frame = obs_arr[frame_t]
            frame = torch.from_numpy(frame).float() / 255.0
            if frame.ndim == 3:
                frame = frame.permute(2, 0, 1)
            return frame

        obs_seq = obs_arr[frame_t : frame_t + self.seq_len]
        act_seq = act_arr[frame_t : frame_t + self.seq_len]
        tgt_seq = obs_arr[frame_t + 1 : frame_t + self.seq_len + 1]

        if fmt == "z":
            # precomputed latents: already float in VAE shape, no rescale / permute
            obs_seq = torch.from_numpy(np.asarray(obs_seq)).float()
            tgt_seq = torch.from_numpy(np.asarray(tgt_seq)).float()
        else:
            obs_seq = torch.from_numpy(obs_seq).float() / 255.0
            tgt_seq = torch.from_numpy(tgt_seq).float() / 255.0
            if obs_seq.ndim == 4:
                obs_seq = obs_seq.permute(0, 3, 1, 2)
                tgt_seq = tgt_seq.permute(0, 3, 1, 2)

        act_seq = torch.from_numpy(act_seq).float()

        if self.one_hot_act:
            a = act_seq.long()
            n_classes = int(a.max().item()) + 1
            one_hot = torch.zeros(*a.shape, n_classes)
            one_hot.scatter_(-1, a.unsqueeze(-1), 1.0)
            act_seq = one_hot

        if fmt == "z":
            # pair (S, 2, *latent_shape): [0]=mu, [1]=logvar
            mu_seq = obs_seq[:, 0]
            logvar_seq = obs_seq[:, 1]
            tgt_mu = tgt_seq[:, 0]
            tgt_logvar = tgt_seq[:, 1]
        else:
            # raw frames: encode on the fly is NOT supported; return as-is
            # (caller must run precompute_z first for the predictor)
            mu_seq = obs_seq
            logvar_seq = torch.zeros_like(obs_seq)
            tgt_mu = tgt_seq
            tgt_logvar = torch.zeros_like(tgt_seq)

        reward_seq = None
        if reward is not None:
            rw = reward[frame_t + 1 : frame_t + self.seq_len + 1]
            reward_seq = torch.from_numpy(np.asarray(rw)).float()

        return mu_seq, logvar_seq, act_seq, tgt_mu, tgt_logvar, reward_seq
