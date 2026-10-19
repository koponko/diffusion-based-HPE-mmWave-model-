"""Global-only mmDiff-style radar-conditioned pose diffusion.

Architecture:
    radar point clouds (B,T,P,C)
        -> PointNet++/MRG encoder
        -> L2 spatial radar tokens (B,T*S,D)
        -> Global Radar Context transformer
        -> joint condition C_global (B,num_joints,64)
        -> pose-space diffusion denoiser

This intentionally implements ONLY the Global Radar Context (GRC).
Local Radar Context, Temporal Motion Consistency, and limb conditioning are
not included.

The PointNet++ encoder must support:
    model(radar, return_aux=True)

and return aux["stage2_features"] shaped (B,T,S,D).
The main encoder output remains (B,T,D).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Basic graph utilities
# -----------------------------------------------------------------------------

# SMPL-X 22 body joint kinematic tree edges (standard 22-joint body skeleton)
# Joint indices in SMPL-X / M4Human dataset:
#   0: Pelvis
#   1: L_Hip,       2: R_Hip,       3: Spine1
#   4: L_Knee,      5: R_Knee,      6: Spine2
#   7: L_Ankle,     8: R_Ankle,     9: Spine3
#  10: L_Foot,     11: R_Foot,     12: Neck
#  13: L_Collar,   14: R_Collar,   15: Head
#  16: L_Shoulder, 17: R_Shoulder, 18: L_Elbow
#  19: R_Elbow,    20: L_Wrist,    21: R_Wrist
SMPLX_22_POSE_EDGES = (
    # Spine and head chain
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left leg chain
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg chain
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Left arm chain
    (9, 13), (13, 16), (16, 18), (18, 20),
    # Right arm chain
    (9, 14), (14, 17), (17, 19), (19, 21),
)

# Human3.6M 17-joint skeleton edges (provided for backwards-compatibility)
H36M_17_POSE_EDGES = (
    (0, 1), (1, 2), (2, 3),          # right leg
    (0, 4), (4, 5), (5, 6),          # left leg
    (0, 7), (7, 8), (8, 9), (9, 10), # spine, thorax, neck, head
    (8, 11), (11, 12), (12, 13),      # left arm
    (8, 14), (14, 15), (15, 16),      # right arm
)

# Default pose edges for SMPL-X (22 body joints)
POSE_EDGES = SMPLX_22_POSE_EDGES


def make_normalized_adjacency(
    num_joints: int = 22,
    edges: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> torch.Tensor:
    """Return symmetric normalized adjacency with self-loops.

    If edges is None, automatically selects SMPLX_22_POSE_EDGES for 22 joints,
    H36M_17_POSE_EDGES for 17 joints, or falls back to POSE_EDGES.
    """
    if edges is None:
        if num_joints == 22:
            edges = SMPLX_22_POSE_EDGES
        elif num_joints == 17:
            edges = H36M_17_POSE_EDGES
        else:
            edges = POSE_EDGES

    adj = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for a, b in edges:
        if a < num_joints and b < num_joints:
            adj[a, b] = 1.0
            adj[b, a] = 1.0
    adj += torch.eye(num_joints, dtype=torch.float32)
    degree = adj.sum(dim=1)
    inv_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)
    return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]


class GraphConv(nn.Module):
    """Simple dense graph convolution: A X W."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B,num_joints,C), adj: (num_joints,num_joints)
        x = torch.einsum("ij,bjc->bic", adj, x)
        return self.linear(x)


# -----------------------------------------------------------------------------
# Global Radar Context
# -----------------------------------------------------------------------------

class GlobalRadarContext(nn.Module):
    """Paper-inspired Global Radar Context (GRC).

    The PointNet++ stage-2 regional features are treated as radar tokens.
    The module prepends trainable joint tokens, applies a transformer over
    joint + radar tokens, and retains the joint outputs.

    Output:
        joint_features: (B,num_joints,grc_dim)
        c_global:       (B,num_joints,condition_dim)
        coarse_pose:    (B,num_joints,3)
    """

    def __init__(
        self,
        radar_feature_dim: int,
        grc_dim: int = 256,
        condition_dim: int = 64,
        num_joints: int = 22,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        temporal_window: int = 4,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.grc_dim = grc_dim
        self.temporal_window = temporal_window

        self.radar_proj = nn.Sequential(
            nn.LayerNorm(radar_feature_dim),
            nn.Linear(radar_feature_dim, grc_dim),
        )

        # Trainable joint feature template + joint positional embedding,
        # following the paper's CLS-token-style construction.
        self.joint_template = nn.Parameter(torch.randn(num_joints, grc_dim) * 0.02)
        self.joint_pos = nn.Parameter(torch.randn(num_joints, grc_dim) * 0.02)

        # Preserve temporal identity of PointNet++ tokens without adding a
        # separate temporal-context module.
        self.frame_pos = nn.Parameter(torch.randn(temporal_window, grc_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=grc_dim,
            nhead=num_heads,
            dim_feedforward=grc_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.condition_projector = nn.Sequential(
            nn.LayerNorm(grc_dim),
            nn.Linear(grc_dim, grc_dim),
            nn.GELU(),
            nn.Linear(grc_dim, condition_dim),
        )

        # Phase-1 coarse pose estimator. This is intentionally separate from
        # the diffusion denoiser.
        self.coarse_pose_head = nn.Sequential(
            nn.LayerNorm(grc_dim),
            nn.Linear(grc_dim, grc_dim // 2),
            nn.GELU(),
            nn.Linear(grc_dim // 2, 3),
        )

    def forward(
        self,
        radar_tokens: torch.Tensor,
        temporal_window: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            radar_tokens:
                (B,T,S,D) PointNet++ spatial features.

        Returns:
            joint_features: (B,num_joints,grc_dim)
            c_global:       (B,num_joints,condition_dim)
            coarse_pose:     (B,num_joints,3)
        """
        if radar_tokens.ndim != 4:
            raise ValueError(
                "radar_tokens must have shape (B,T,S,D), "
                f"got {tuple(radar_tokens.shape)}"
            )

        batch, frames, regions, _ = radar_tokens.shape
        if temporal_window is None:
            temporal_window = frames
        if temporal_window > self.frame_pos.shape[0]:
            raise ValueError(
                f"temporal_window={temporal_window} exceeds configured "
                f"maximum {self.frame_pos.shape[0]}"
            )

        radar = self.radar_proj(radar_tokens)
        radar = radar + self.frame_pos[:frames][None, :, None, :]
        radar = radar.reshape(batch, frames * regions, self.grc_dim)

        joint_tokens = (
            self.joint_template + self.joint_pos
        )[None, :, :].expand(batch, -1, -1)

        tokens = torch.cat([joint_tokens, radar], dim=1)
        encoded = self.transformer(tokens)

        joint_features = encoded[:, : self.num_joints]
        c_global = self.condition_projector(joint_features)
        coarse_pose = self.coarse_pose_head(joint_features)

        return joint_features, c_global, coarse_pose


# -----------------------------------------------------------------------------
# PointNet++ -> GRC adapter
# -----------------------------------------------------------------------------

class GlobalRadarConditionEncoder(nn.Module):
    """Wrap an existing PointNetPPFeatureExtractor and expose global context."""

    def __init__(
        self,
        pointnet_encoder: nn.Module,
        pointnet_feature_dim: int = 256,
        condition_dim: int = 64,
        temporal_window: int = 4,
        grc_dim: int = 256,
        grc_layers: int = 4,
        grc_heads: int = 8,
        dropout: float = 0.1,
        num_joints: int = 22,
    ):
        super().__init__()
        self.pointnet = pointnet_encoder
        self.num_joints = num_joints
        self.grc = GlobalRadarContext(
            radar_feature_dim=pointnet_feature_dim,
            grc_dim=grc_dim,
            condition_dim=condition_dim,
            num_joints=num_joints,
            temporal_window=temporal_window,
            num_layers=grc_layers,
            num_heads=grc_heads,
            dropout=dropout,
        )

    def forward(self, radar: torch.Tensor) -> Dict[str, torch.Tensor]:
        # The encoder's default output remains the per-frame global feature,
        # but GRC consumes the spatial L2 features from return_aux.
        _, aux = self.pointnet(
            radar,
            return_aux=True,
        )
        if "stage2_features" not in aux:
            raise RuntimeError(
                "PointNetPPFeatureExtractor must return aux['stage2_features'] "
                "with shape (B,T,S,D)."
            )

        stage2 = aux["stage2_features"]
        joint_features, c_global, coarse_pose = self.grc(stage2)

        return {
            "radar_features": stage2,
            "joint_features": joint_features,
            "c_global": c_global,
            "coarse_pose": coarse_pose,
        }


# -----------------------------------------------------------------------------
# Pose diffusion denoiser: GRC-only conditioning
# -----------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 2:
            raise ValueError("time embedding dimension must be >= 2")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = -torch.log(
            torch.tensor(10000.0, device=t.device, dtype=torch.float32)
        ) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * exponent
        )
        phase = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        emb = torch.cat([phase.sin(), phase.cos()], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalGraphResBlock(nn.Module):
    """Residual GCN block with time and global-radar conditioning."""

    def __init__(self, hidden_dim: int, condition_dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gconv1 = GraphConv(hidden_dim, hidden_dim)
        self.gconv2 = GraphConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        condition: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        h = self.norm1(x)
        h = self.gconv1(h, adj)
        h = h + self.time_proj(t_emb)[:, None, :]
        h = h + self.condition_proj(condition)
        h = F.gelu(h)
        h = self.dropout(h)

        h = self.norm2(h)
        h = self.gconv2(h, adj)
        h = self.dropout(h)
        return residual + h


class GlobalConditionedPoseDiffusion(nn.Module):
    """Pose-space epsilon predictor conditioned only on GRC."""

    def __init__(
        self,
        condition_dim: int = 64,
        hidden_dim: int = 96,
        time_dim: int = 96,
        num_blocks: int = 6,
        dropout: float = 0.1,
        num_joints: int = 22,
        edges: Optional[Tuple[Tuple[int, int], ...]] = None,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        adj = make_normalized_adjacency(num_joints, edges=edges)
        self.register_buffer("adj", adj, persistent=False)

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.pose_input = GraphConv(3, hidden_dim)
        self.condition_input = nn.Linear(condition_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ConditionalGraphResBlock(
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                time_dim=time_dim,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pose_output = GraphConv(hidden_dim, 3)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:      (B,num_joints,3) noisy pose.
            t:        (B,) diffusion timestep.
            c_global: (B,num_joints,condition_dim) global radar condition.

        Returns:
            epsilon_pred: (B,num_joints,3)
        """
        if x_t.ndim != 3 or x_t.shape[1:] != (self.num_joints, 3):
            raise ValueError(
                f"x_t must be (B,{self.num_joints},3), got {tuple(x_t.shape)}"
            )
        if c_global.shape[:2] != x_t.shape[:2]:
            raise ValueError(
                "c_global must have the same batch/joint dimensions as x_t"
            )

        t_emb = self.time_embedding(t)
        h = self.pose_input(x_t, self.adj)
        h = h + self.condition_input(c_global)

        for block in self.blocks:
            h = block(h, t_emb, c_global, self.adj)

        return self.pose_output(self.output_norm(h), self.adj)


# -----------------------------------------------------------------------------
# Diffusion training utilities
# -----------------------------------------------------------------------------

class ConstantBetaSchedule:
    """Small schedule matching the paper's stated 25-step, beta=0.001 setup."""

    def __init__(self, num_steps: int = 25, beta: float = 1e-3):
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0,1)")

        self.num_steps = num_steps
        betas = torch.full((num_steps,), float(beta))
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def to(self, device: torch.device) -> "ConstantBetaSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        x_t = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise
        return x_t, noise


@torch.no_grad()
def compute_coarse_context(
    context_encoder: GlobalRadarConditionEncoder,
    radar: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Convenience wrapper for phase-1 outputs."""
    context_encoder.eval()
    return context_encoder(radar)


# -----------------------------------------------------------------------------
# DDIM intentionally left for the project owner to implement.
# -----------------------------------------------------------------------------


def ddim_step_formula_note() -> str:
    return (
        "Given epsilon_theta(x_t,t,C), compute x0_hat = "
        "(x_t - sqrt(1-a_t)*eps)/sqrt(a_t). Then use alpha_bar_t and "
        "alpha_bar_prev to construct the DDIM sigma and x_{t-1}. For eta=0 "
        "the update is deterministic. Start from the coarse pose H~ in this "
        "paper-specific setup rather than pure Gaussian noise."
    )


if __name__ == "__main__":
    print(ddim_step_formula_note())
