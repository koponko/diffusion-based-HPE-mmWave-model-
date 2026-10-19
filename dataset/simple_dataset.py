"""
Lightweight adapter dataset for M4Human RF3DPoseDataset.

Provides a simplified output suitable for experimenting with diffusion models:
- "radar_points": torch.FloatTensor of shape (T, P, C) (C>=3)
- "joints": torch.FloatTensor of shape (J, 3) in meters
- "vertices": optional torch.FloatTensor of shape (V, 3) in meters
- "indicator": original indicator tuple (subject, action, frame)

This adapter normalizes units so joints and vertices are in meters (SMPL-X in this repo uses mm for vertices).
"""

from typing import Optional
import torch
from torch.utils.data import Dataset

import numpy as np

try:
    # use the existing RF3DPoseDataset from this repo when available
    from .m4human_dataset import RF3DPoseDataset
except Exception:
    RF3DPoseDataset = None


class SimpleM4HumanDataset(Dataset):
    """Adapter that exposes a minimal interface for model development.

    Returns a dict with keys: 'radar_points', 'joints', optional 'vertices', and 'indicator'.
    """

    def __init__(self, rf3d_dataset, transform: Optional[callable] = None):
        if RF3DPoseDataset is not None and isinstance(rf3d_dataset, RF3DPoseDataset):
            self.ds = rf3d_dataset
        elif isinstance(rf3d_dataset, (list, tuple)):
            # allow passing a pre-built list of samples (for unit tests / small examples)
            self.ds = rf3d_dataset
        else:
            raise ValueError("rf3d_dataset must be an RF3DPoseDataset instance or a list/tuple of samples")

        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        # radar_points: expected shape (T, P, C) with C>=3
        radar_points = sample.get("radar_points")
        if radar_points is None:
            raise KeyError("Underlying dataset must provide 'radar_points' key")

        # joints: prefer 'joints_root' (SMPL-X derived joint positions in meters)
        joints = sample.get("joints_root")
        if joints is None:
            joints = sample.get("joints")

        # vertices: SMPL-X vertices returned by generate_vertices_in_radar_space() are in millimeters in this repo
        vertices = sample.get("vertices")

        # If joints missing, try to derive a proxy from vertices
        if joints is None and vertices is not None:
            v = np.asarray(vertices)
            # simplest proxy: take first 22 vertices (heuristic) and convert to meters
            if v.ndim == 2 and v.shape[0] >= 22:
                joints = v[:22] / 1000.0
            elif v.ndim == 3 and v.shape[1] >= 22:
                joints = v[-1][:22] / 1000.0
            else:
                joints = None

        if joints is None:
            raise KeyError("Underlying dataset must provide 'joints_root' or 'vertices' to derive joints")

        # Convert to torch tensors with consistent dtypes
        radar_pts_t = torch.from_numpy(np.asarray(radar_points)).float()
        joints_t = torch.from_numpy(np.asarray(joints)).float()

        vertices_t = None
        if vertices is not None:
            v_arr = np.asarray(vertices).astype(np.float32)
            # convert mm -> m if necessary
            if v_arr.max() > 50.0:
                v_arr = v_arr / 1000.0
            vertices_t = torch.from_numpy(v_arr).float()

        out = {
            "radar_points": radar_pts_t,  # (T, P, C)
            "joints": joints_t,  # (J, 3) in meters
            "vertices": vertices_t,  # optional (V, 3) in meters
            "indicator": sample.get("indicator"),
        }

        if self.transform is not None:
            out = self.transform(out)

        return out