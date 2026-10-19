"""
Minimal diffusion-style base model for point-cloud / skeleton sequences.

This is intentionally simple to serve as a starting point:
- a sinusoidal time embedding
- a small MLP that consumes flattened (T * P * C) inputs + time embedding
- forward(x, t) predicts the same-shaped noise tensor as input

Intended usage:
    model = DiffusionPointCloudModel(temporal_window=4, num_points=1000)
    x = torch.randn(2, 4, 1000, 3)  # batch of noisy point-cloud sequences
    t = torch.randint(0, 1000, (2,))  # timesteps
    out = model(x, t)  # prediction of noise, same shape as x

The model is small and not optimized for performance. Replace the MLP with a Transformer or PointNet-style encoder later
when moving beyond this prototype.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extract import PointNetPPFeatureExtractor


def sinusoidal_time_embedding(timesteps, dim: int):
    """Create sinusoidal embeddings for a batch of timesteps.

    timesteps: Tensor of shape (B,) with integer timestep values
    returns: Tensor (B, dim)
    """
    assert timesteps.dim() == 1
    device = timesteps.device
    dtype = timesteps.dtype
    half = dim // 2
    # construct the frequency terms on the same device/dtype as timesteps
    log_term = torch.log(torch.tensor(10000.0, device=device, dtype=torch.float32)) / (half - 1)
    freqs = torch.exp(torch.arange(half, dtype=torch.float32, device=device) * -log_term)
    emb = timesteps.to(torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # project sinusoidal embedding into a learned space
        self.proj = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor):
        # t: (B,) integer timesteps
        emb = sinusoidal_time_embedding(t, self.dim).to(t.device)
        return self.proj(emb)


class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DiffusionPointCloudModel(nn.Module):
    """Very small diffusion backbone.

    Predicts noise for input point-cloud sequences.

    Parameters:
    - in_channels: typically 3 (x,y,z)
    - temporal_window: T
    - num_points: max number of points per frame P (inputs are expected to be padded/truncated to this)
    - hidden_dim: intermediate feature size
    - time_emb_dim: dimension of time embedding
    """

    def __init__(
        self,
        in_channels: int = 3,
        temporal_window: int = 4,
        num_points: int = 1000,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.T = temporal_window
        self.P = num_points
        self.input_dim = self.T * self.P * self.in_channels
        self.hidden_dim = hidden_dim

        # time embedding
        self.time_emb = TimeEmbedding(time_emb_dim)
        # project time embedding to hidden and then broadcast
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        self.encoder = PointNetPPFeatureExtractor(
            in_channels=in_channels,
            feature_dim=hidden_dim,
            first_sample_count=min(256, num_points),
            second_sample_count=min(64, num_points),
        )

        # residual MLP blocks
        self.blocks = nn.ModuleList([MLPBlock(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(2)])

        # Predict noise independently at each point, conditioned on frame/global features.
        self.final = nn.Sequential(
            nn.Linear(in_channels + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_channels),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """Forward pass.

        x: (B, T, P, C)
        t: (B,) integer timesteps

        returns: Tensor same shape as x with noise prediction
        """
        B = x.shape[0]
        assert x.ndim == 4, "x must be (B, T, P, C)"
        assert x.shape[1] == self.T, f"Expected temporal window {self.T}, got {x.shape[1]}"
        frame_features = self.encoder(x)

        t_emb = self.time_emb(t)
        t_proj = self.time_proj(t_emb)
        time_features = t_proj.unsqueeze(1).expand(-1, self.T, -1)
        conditioning = frame_features + time_features

        for block in self.blocks:
            conditioning = conditioning + block(conditioning)

        point_conditioning = conditioning.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
        out = self.final(torch.cat([x, point_conditioning], dim=-1))
        return out


if __name__ == "__main__":
    # quick sanity check
    m = DiffusionPointCloudModel(in_channels=3, temporal_window=2, num_points=16, hidden_dim=128)
    x = torch.randn(2, 2, 16, 3)
    t = torch.randint(0, 1000, (2,))
    y = m(x, t)
    print("Input shape:", x.shape, "Output shape:", y.shape)
