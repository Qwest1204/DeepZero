import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset


def _find_sessions(data_dir, game):
    obs_act_pairs = []

    if game in ("car", None):
        act_files = glob.glob(os.path.join(data_dir, "actions-car*.npy"))
        for act_path in act_files:
            m = re.search(r"actions-car(\d+)\.npy", os.path.basename(act_path))
            if not m:
                continue
            session_id = m.group(1)
            obs_path = os.path.join(data_dir, f"observations-car{session_id}.npy")
            if os.path.exists(obs_path):
                obs_act_pairs.append((obs_path, act_path))

    if game in ("doom", None):
        act_files = glob.glob(os.path.join(data_dir, "doom-act*.npy"))
        for act_path in act_files:
            m = re.search(r"doom-act(\d+)\.npy", os.path.basename(act_path))
            if not m:
                continue
            session_id = m.group(1)
            obs_path = os.path.join(data_dir, f"doom-obs{session_id}.npy")
            if os.path.exists(obs_path):
                obs_act_pairs.append((obs_path, act_path))

    if not obs_act_pairs:
        available = glob.glob(os.path.join(data_dir, "*.npy"))
        names = [os.path.basename(p) for p in available]
        raise FileNotFoundError(
            f"Не найдено пар obs+act в '{data_dir}'.\n"
            f"Искал: actions-car*.npy + observations-car*.npy "
            f"или doom-act*.npy + doom-obs*.npy.\n"
            f"Найденные .npy файлы: {names if names else '(нет .npy)'}"
        )

    return obs_act_pairs


class RecordingDataset(Dataset):
    def __init__(self, data_dir="try", game=None, seq_len=8, mode="predictor"):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.mode = mode

        pairs = _find_sessions(data_dir, game)

        self.sessions = []
        for obs_path, act_path in pairs:
            obs_arr = np.load(obs_path)
            act_arr = np.load(act_path)

            n_obs = len(obs_arr)
            n_act = len(act_arr)

            if n_obs == n_act + 1:
                fmt = "car"
            elif n_obs == n_act:
                fmt = "doom"
            else:
                raise ValueError(
                    f"{os.path.basename(obs_path)}: len(obs)={n_obs}, "
                    f"len(act)={n_act}. Ожидалось n_obs=n_act+1 (CarRacing) "
                    f"или n_obs=n_act (Doom)."
                )

            self.sessions.append((obs_arr, act_arr, fmt))

        self._build_index()

    def _build_index(self):
        cumlens = []
        valid_sessions = []

        for obs, act, fmt in self.sessions:
            if self.mode == "vae":
                n_samples = len(obs)
            elif fmt == "car":
                n_samples = len(obs) - self.seq_len - 1
            else:
                n_samples = len(obs) - self.seq_len

            if n_samples > 0:
                prev = cumlens[-1] if cumlens else 0
                cumlens.append(prev + n_samples)
                valid_sessions.append((obs, act, n_samples))

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
        obs_arr, act_arr = self.sessions[session_idx][:2]

        if self.mode == "vae":
            frame = obs_arr[frame_t]
            frame = torch.from_numpy(frame).float() / 255.0
            if frame.ndim == 3:
                frame = frame.permute(2, 0, 1)
            return frame

        obs_seq = obs_arr[frame_t : frame_t + self.seq_len]
        act_seq = act_arr[frame_t : frame_t + self.seq_len]
        tgt_seq = obs_arr[frame_t + 1 : frame_t + self.seq_len + 1]

        obs_seq = torch.from_numpy(obs_seq).float() / 255.0
        tgt_seq = torch.from_numpy(tgt_seq).float() / 255.0

        if obs_seq.ndim == 4:
            obs_seq = obs_seq.permute(0, 3, 1, 2)
            tgt_seq = tgt_seq.permute(0, 3, 1, 2)

        act_seq = torch.from_numpy(act_seq).float()
        if act_seq.ndim == 2 and act_seq.shape[-1] == 4:
            act_seq = act_seq[:, :3]

        return obs_seq, act_seq, tgt_seq