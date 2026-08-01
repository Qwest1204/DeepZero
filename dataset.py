import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset


def _glob_both(data_dir, pattern):
    """Glob in data_dir and its Doom/ and MW/ subdirectories."""
    return (
        glob.glob(os.path.join(data_dir, pattern))
        + glob.glob(os.path.join(data_dir, "Doom", pattern))
        + glob.glob(os.path.join(data_dir, "MW", pattern))
    )


def _find_sessions(data_dir, game):
    obs_act_pairs = []

    if game in ("car", None):
        act_files = _glob_both(data_dir, "actions-car*.npy")
        for act_path in act_files:
            m = re.search(r"actions-car(\d+)\.npy", os.path.basename(act_path))
            if not m:
                continue
            session_id = m.group(1)
            obs_path = os.path.join(os.path.dirname(act_path), f"observations-car{session_id}.npy")
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
            f"Искал: actions-car*.npy + observations-car*.npy, "
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


def precompute_z(data_dir="try", vae=None, device="cpu", camera_idx=1,
                 dtype=np.float16):
    """Encode subsampled MetaWorld frames with a frozen VAE and interpolate gaps.

    For every ``metaworld-obs{idx}-{task}.npy`` session:
        1. picks camera ``camera_idx``,
        2. encodes the saved (every-2nd) frames with the VAE mean,
        3. linearly interpolates the missing odd latents,
        4. saves the full latent sequence to ``metaworld-z{idx}-{task}.npy``
           next to the obs file.

    Args:
        data_dir: Root data directory (searches ``MW/`` subdirectory too).
        vae: Frozen VAE instance (``encode(x) -> (mu, logvar)``).
        device: Torch device for encoding.
        camera_idx: Camera index inside the saved (C, H, W, 3) obs axis.
        dtype: Numpy dtype for the saved latents.

    Returns:
        List of written ``metaworld-z*.npy`` paths.
    """
    if vae is None:
        raise ValueError("precompute_z требует vae (from_pretrained VAE)")
    vae = vae.to(device)
    vae.eval()

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
        z_path = os.path.join(
            os.path.dirname(act_path), f"metaworld-z{session_id}-{task}.npy"
        )
        if not os.path.exists(obs_path):
            continue
        obs_arr = np.load(obs_path)
        if obs_arr.ndim == 5:
            obs_arr = obs_arr[:, camera_idx]

        x = torch.from_numpy(obs_arr).float().to(device) / 255.0
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        with torch.no_grad():
            mu, _ = vae.encode(x)
        z_even = mu.detach().cpu().numpy().astype(dtype)
        z_full = interp_even_latents(z_even)
        np.save(z_path, z_full)
        written.append(z_path)
        print(f"precompute_z: {os.path.basename(obs_path)} "
              f"{obs_arr.shape} -> {z_full.shape} ({z_full.nbytes / 1e6:.1f} MB)")

    if not written:
        print("precompute_z: не найдено сессий metaworld-obs*.npy")
    return written


class RecordingDataset(Dataset):
    def __init__(self, data_dir="try", game=None, seq_len=8, mode="predictor",
                 camera_idx=1):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.mode = mode
        self.camera_idx = camera_idx

        pairs = _find_sessions(data_dir, game)

        self.sessions = []
        for obs_path, act_path in pairs:
            act_arr = np.load(act_path)

            # MetaWorld: prefer precomputed full latents for the predictor.
            z_path = None
            if "metaworld" in os.path.basename(obs_path):
                m = re.search(r"metaworld-obs(\d+)-(.*)\.npy", os.path.basename(obs_path))
                if m and mode == "predictor":
                    candidate = os.path.join(
                        os.path.dirname(obs_path),
                        f"metaworld-z{m.group(1)}-{m.group(2)}.npy",
                    )
                    if os.path.exists(candidate):
                        z_path = candidate

            if z_path is not None:
                obs_arr = np.load(z_path)
                fmt = "metaworld_z"
            else:
                obs_arr = np.load(obs_path)
                # MetaWorld multi-camera obs: (N, C, H, W, 3) -> select one camera
                if obs_arr.ndim == 5:
                    obs_arr = obs_arr[:, self.camera_idx]

            n_obs = len(obs_arr)
            n_act = len(act_arr)

            if z_path is not None:
                if n_obs not in (n_act, n_act - 1):
                    raise ValueError(
                        f"{os.path.basename(obs_path)}: precomputed z len={n_obs}, "
                        f"len(act)={n_act}. Ожидалось n_obs=n_act или n_obs=n_act-1."
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
            if fmt in ("metaworld", "metaworld_z"):
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

            self.sessions.append((obs_arr, act_arr, fmt, done_arr, subsampled, reward_arr))

        self._build_index()

    def _build_index(self):
        cumlens = []
        valid_sessions = []

        for obs, act, fmt, done, subsampled, reward in self.sessions:
            if self.mode == "vae":
                allowed = np.arange(len(obs))
            else:
                n_samples = len(obs) - self.seq_len - 1 if fmt == "car" else len(obs) - self.seq_len
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

        if fmt == "metaworld_z":
            # precomputed latents: already float, no rescale / channel permute
            obs_seq = torch.from_numpy(np.asarray(obs_seq)).float()
            tgt_seq = torch.from_numpy(np.asarray(tgt_seq)).float()
        else:
            obs_seq = torch.from_numpy(obs_seq).float() / 255.0
            tgt_seq = torch.from_numpy(tgt_seq).float() / 255.0
            if obs_seq.ndim == 4:
                obs_seq = obs_seq.permute(0, 3, 1, 2)
                tgt_seq = tgt_seq.permute(0, 3, 1, 2)

        act_seq = torch.from_numpy(act_seq).float()
        if fmt == "car" and act_seq.ndim == 2 and act_seq.shape[-1] == 4:
            act_seq = act_seq[:, :3]

        return obs_seq, act_seq, tgt_seq
