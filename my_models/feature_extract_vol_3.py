"""PointNet++-style feature extraction for temporal mmWave radar point clouds."""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------

class SharedMLP(nn.Module):
    """Pointwise MLP applied independently to the last feature dimension."""

    def __init__(self, in_channels, channels):
        super().__init__()

        if isinstance(channels, int):
            channels = [channels]

        layers = []
        current = in_channels

        for channel in channels:
            layers.extend([
                nn.Linear(current, channel),
                nn.LayerNorm(channel),
                nn.SiLU(),
            ])
            current = channel

        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)


class PointwiseScoreMLP(nn.Module):
    """Pointwise MLP whose final layer produces an unrestricted scalar score."""

    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, features):
        return self.net(features)


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

def farthest_point_sample(points, sample_count):
    """Select point indices with farthest-point sampling.

    Args:
        points: Tensor shaped (B, N, 3).
        sample_count: Number of points to select.

    Returns:
        LongTensor shaped (B, sample_count).
    """
    batch_size, point_count, _ = points.shape

    sample_count = min(sample_count, point_count)

    indices = torch.empty(
        batch_size,
        sample_count,
        dtype=torch.long,
        device=points.device,
    )

    distances = torch.full(
        (batch_size, point_count),
        float("inf"),
        device=points.device,
    )

    farthest = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=points.device,
    )

    batch_indices = torch.arange(
        batch_size,
        device=points.device,
    )

    for sample_index in range(sample_count):
        indices[:, sample_index] = farthest

        center = points[
            batch_indices,
            farthest
        ].unsqueeze(1)

        distance = (
            (points - center) ** 2
        ).sum(dim=-1)

        distances = torch.minimum(
            distances,
            distance,
        )

        farthest = distances.max(dim=-1).indices

    return indices


def random_point_sample(points, sample_count):
    """Select unique random point indices independently for each batch item.

    Args:
        points: Tensor shaped (B, N, 3).
        sample_count: Number of points to select.

    Returns:
        LongTensor shaped (B, sample_count).
    """
    batch_size, point_count, _ = points.shape
    sample_count = min(sample_count, point_count)

    indices = torch.stack([
        torch.randperm(
            point_count,
            device=points.device,
        )[:sample_count]
        for _ in range(batch_size)
    ])

    return indices


def index_points(points, indices):
    """Gather batched points/features with indices shaped (B, ...).

    Args:
        points: (B, N, C)
        indices: (B, ...)

    Returns:
        (B, ..., C)
    """
    batch_indices = torch.arange(
        points.shape[0],
        device=points.device,
    )

    view_shape = [points.shape[0]] + [1] * (indices.ndim - 1)

    batch_indices = (
        batch_indices
        .view(*view_shape)
        .expand_as(indices)
    )

    return points[batch_indices, indices]


# ---------------------------------------------------------------------
# PointNet T-Net
# ---------------------------------------------------------------------

class TransformNet(nn.Module):
    """PointNet-style T-Net predicting a square feature transform."""

    def __init__(
        self,
        in_channels,
        transform_dim,
        hidden_channels=(64, 128, 256),
    ):
        super().__init__()

        self.transform_dim = transform_dim

        self.encoder = SharedMLP(
            in_channels,
            hidden_channels,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_channels[-1], 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, transform_dim * transform_dim),
        )

        # Initialize the transform close to identity.
        final_linear = self.head[-1]

        nn.init.zeros_(final_linear.weight)

        identity = torch.eye(transform_dim).reshape(-1)

        with torch.no_grad():
            final_linear.bias.copy_(identity)

    def forward(self, features):
        """
        Args:
            features: (B, N, C)

        Returns:
            transform: (B, transform_dim, transform_dim)
        """
        encoded = self.encoder(features)

        # Global PointNet-style aggregation.
        global_feature = encoded.max(dim=1).values

        transform = self.head(global_feature)

        transform = transform.view(
            features.shape[0],
            self.transform_dim,
            self.transform_dim,
        )

        return transform


def transform_regularizer(transform):
    """Orthogonality regularizer used by PointNet T-Nets.

    Encourages A A^T to remain close to the identity matrix.
    """
    batch_size, channels, _ = transform.shape

    identity = torch.eye(
        channels,
        device=transform.device,
        dtype=transform.dtype,
    ).unsqueeze(0).expand(batch_size, -1, -1)

    product = transform @ transform.transpose(1, 2)

    return ((product - identity) ** 2).mean()


# ---------------------------------------------------------------------
# Neighborhood grouping
# ---------------------------------------------------------------------

def knn_group(
    points,
    centers,
    neighbor_count,
    radius_scale=2.0,
):
    """Find each center's k nearest neighbors and a density-aware local scale.

    Args:
        points: (B, N, 3)
        centers: (B, S, 3)
        neighbor_count: K, number of neighbors to gather per center.
        radius_scale: multiplier applied to the per-center local distance
            scale (median neighbor distance) to get a "local_radius" that
            is not used to filter neighbors here, but is returned so callers
            can use it to normalize local geometry against point-density
            changes between frames.

    Returns:
        indices:          (B, S, K) indices into `points`, nearest first.
        local_radius:      (B, S) density-aware normalization scale.
        neighbor_distances: (B, S, K) Euclidean distance to each neighbor,
            index-aligned with `indices`.
    """

    batch_size, point_count, _ = points.shape

    neighbor_count = min(
        neighbor_count,
        point_count,
    )

    distances = torch.cdist(
        centers,
        points,
    )

    base_distances, base_indices = distances.topk(
        neighbor_count,
        dim=-1,
        largest=False,
    )

    if neighbor_count == 1:
        local_scale = base_distances[..., 0]
    else:
        # Ignore the zero/self-distance when possible.
        local_scale = base_distances[..., 1:].median(
            dim=-1
        ).values

    local_radius = (
        radius_scale
        * local_scale.clamp_min(1e-6)
    )

    return (
        base_indices,
        local_radius,
        base_distances,
    )


# ---------------------------------------------------------------------
# PointNet++ Set Abstraction
# ---------------------------------------------------------------------

class PointNetSetAbstraction(nn.Module):
    """PointNet++ set abstraction with robust grouping and optional hybrid sampling.

    Sampling modes:

        "fps":
            Standard farthest point sampling.

        "random":
            Pure random point sampling.

        "hybrid":
            Generate FPS and random candidates and learn how much the
            sampler should prefer the random candidates.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        sample_count,
        neighbor_count,
        sampling_mode="fps",
        robust_grouping=True,
        radius_scale=2.0,
    ):
        super().__init__()

        if sampling_mode not in {"fps", "random", "hybrid"}:
            raise ValueError(
                "sampling_mode must be 'fps', 'random', or 'hybrid'"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_count = sample_count
        self.neighbor_count = neighbor_count
        self.sampling_mode = sampling_mode
        self.robust_grouping = robust_grouping
        self.radius_scale = radius_scale

        # The MLP lives after grouping and operates on:
        #
        #       relative_xyz (3) + point features (in_channels)
        #       + neighbor distance (1)
        #
        self.mlp = SharedMLP(
            in_channels + 3 + 1,
            out_channels,
        )

        # Learned hybrid sampler.
        if sampling_mode == "hybrid":
            candidate_channels = in_channels + 3

            self.sampler_score = PointwiseScoreMLP(
                candidate_channels,
                hidden_channels=32,
            )

            context_channels = in_channels + 3

            self.random_ratio_head = nn.Sequential(
                nn.Linear(context_channels, 32),
                nn.LayerNorm(32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )

            # Start slightly closer to FPS than random.
            nn.init.constant_(
                self.random_ratio_head[-1].bias,
                -1.1,
            )

    def _select_centers(self, points, features):
        """Select S centers and optionally return the learned random ratio."""

        sample_count = min(
            self.sample_count,
            points.shape[1],
        )

        # -------------------------------------------------------------
        # Standard FPS
        # -------------------------------------------------------------
        fps_indices = farthest_point_sample(
            points,
            sample_count,
        )

        fps_centers = index_points(
            points,
            fps_indices,
        )

        if self.sampling_mode == "fps":
            return fps_centers, None

        # -------------------------------------------------------------
        # Pure random sampling
        # -------------------------------------------------------------
        if self.sampling_mode == "random":
            random_indices = random_point_sample(
                points,
                sample_count,
            )

            random_centers = index_points(
                points,
                random_indices,
            )

            return random_centers, None

        # -------------------------------------------------------------
        # Learned FPS + random hybrid
        # -------------------------------------------------------------

        if sample_count == points.shape[1]:
            # When S == N there is no reason to create a larger candidate set.
            return fps_centers, torch.zeros(
                points.shape[0],
                1,
                device=points.device,
                dtype=points.dtype,
            )

        random_indices = random_point_sample(
            points,
            sample_count,
        )

        random_centers = index_points(
            points,
            random_indices,
        )

        fps_features = index_points(
            features,
            fps_indices,
        )

        random_features = index_points(
            features,
            random_indices,
        )

        candidate_points = torch.cat(
            [
                fps_centers,
                random_centers,
            ],
            dim=1,
        )

        candidate_features = torch.cat(
            [
                fps_features,
                random_features,
            ],
            dim=1,
        )

        candidate_inputs = torch.cat(
            [
                candidate_points,
                candidate_features,
            ],
            dim=-1,
        )

        # -------------------------------------------------------------
        # Predict a random-vs-FPS preference.
        # -------------------------------------------------------------
        context_xyz = points.mean(dim=1)
        context_features = features.mean(dim=1)

        context = torch.cat(
            [
                context_xyz,
                context_features,
            ],
            dim=-1,
        )

        random_ratio = torch.sigmoid(
            self.random_ratio_head(context)
        ).clamp(
            min=1e-4,
            max=1.0 - 1e-4,
        )

        # -------------------------------------------------------------
        # Score each candidate.
        # -------------------------------------------------------------
        raw_scores = self.sampler_score(
            candidate_inputs
        ).squeeze(-1)

        fps_prior = torch.log1p(
            -random_ratio
        )

        random_prior = torch.log(
            random_ratio
        )

        prior = torch.cat(
            [
                fps_prior.expand(-1, sample_count),
                random_prior.expand(-1, sample_count),
            ],
            dim=1,
        )

        selection_scores = raw_scores + prior

        chosen_indices = selection_scores.topk(
            sample_count,
            dim=1,
            largest=True,
        ).indices

        centers = torch.gather(
            candidate_points,
            dim=1,
            index=chosen_indices.unsqueeze(-1).expand(
                -1,
                -1,
                3,
            ),
        )

        return centers, random_ratio

    def forward(self, points, features):
        """
        Args:
            points:   (B, N, 3)
            features: (B, N, C)

        Returns:
            centers:         (B, S, 3)
            pooled_features: (B, S, out_channels)
            aux:             dictionary with optional sampler information
        """

        # -------------------------------------------------------------
        # 1. Select representative centers
        # -------------------------------------------------------------
        centers, random_ratio = self._select_centers(
            points,
            features,
        )

        # -------------------------------------------------------------
        # 2. Find local neighborhoods
        # -------------------------------------------------------------
        neighbor_count = min(
            self.neighbor_count,
            points.shape[1],
        )

        neighbor_indices, local_scale, neighbor_distances = knn_group(
            points,
            centers,
            neighbor_count,
            self.radius_scale,
        )

        if not self.robust_grouping:
            local_scale = None

        # -------------------------------------------------------------
        # 3. Gather local point coordinates/features
        # -------------------------------------------------------------
        neighbor_points = index_points(
            points,
            neighbor_indices,
        )

        neighbor_features = index_points(
            features,
            neighbor_indices,
        )

        # -------------------------------------------------------------
        # 4. Convert XYZ to coordinates relative to the center
        # -------------------------------------------------------------
        relative_points = (
            neighbor_points
            - centers.unsqueeze(2)
        )

        # Optional density normalization.
        #
        # This makes the local geometry less sensitive to radar point
        # density changing from frame to frame. The neighbor distance
        # feature is normalized by the same scale so it stays consistent
        # with the (now scale-invariant) relative coordinates.
        if local_scale is not None:
            scale = local_scale.clamp_min(1e-6)

            relative_points = (
                relative_points
                / scale.unsqueeze(-1).unsqueeze(-1)
            )

            neighbor_distances = (
                neighbor_distances
                / scale.unsqueeze(-1)
            )

        # -------------------------------------------------------------
        # 5. Relative XYZ + point features + neighbor distance
        # -------------------------------------------------------------
        local_features = torch.cat(
            [
                relative_points,
                neighbor_features,
                neighbor_distances.unsqueeze(-1),
            ],
            dim=-1,
        )

        # -------------------------------------------------------------
        # 6. Shared PointNet MLP
        # -------------------------------------------------------------
        local_features = self.mlp(
            local_features
        )

        # -------------------------------------------------------------
        # 7. Symmetric aggregation over each neighborhood
        # -------------------------------------------------------------
        pooled_features = local_features.max(
            dim=2
        ).values

        aux = {}

        if random_ratio is not None:
            aux["random_ratio"] = random_ratio

        return (
            centers,
            pooled_features,
            aux,
        )


# ---------------------------------------------------------------------
# Optional PointNet++ feature propagation for segmentation
# ---------------------------------------------------------------------

def three_nn_interpolate(
    query_xyz,
    source_xyz,
    source_features,
):
    """Interpolate source features onto query points with 3-NN.

    Args:
        query_xyz:      (B, N, 3)
        source_xyz:     (B, S, 3)
        source_features:(B, S, C)

    Returns:
        interpolated:   (B, N, C)
    """
    k = min(3, source_xyz.shape[1])

    distances = torch.cdist(
        query_xyz,
        source_xyz,
    )

    knn_distances, knn_indices = distances.topk(
        k,
        dim=-1,
        largest=False,
    )

    weights = 1.0 / (
        knn_distances + 1e-6
    )

    weights = (
        weights
        / weights.sum(
            dim=-1,
            keepdim=True,
        )
    )

    grouped_features = index_points(
        source_features,
        knn_indices,
    )

    interpolated = (
        grouped_features
        * weights.unsqueeze(-1)
    ).sum(dim=2)

    return interpolated



class PointNetMRG(nn.Module):
    """
    PointNet++ Multi-Resolution Grouping (MRG).

    At the current level, each region is represented by two vectors:

    1. Lower-resolution branch:
       Aggregate already-computed features from the previous level.

    2. Raw-point branch:
       Directly apply PointNet to raw points in a larger local region.

    The two vectors are concatenated.

    Args:
        lower_feature_channels:
            Number of channels from the previous abstraction level.

        raw_feature_channels:
            Number of channels associated with raw points.

        lower_out_channels:
            Output size of lower-resolution branch.

        raw_out_channels:
            Output size of raw-point branch.

        sample_count:
            Number of current-level centers.

        lower_neighbor_count:
            Number of lower-level features grouped around each center.

        raw_neighbor_count:
            Number of raw points grouped around each center.

        lower_radius:
            Radius for grouping lower-level features.

        raw_radius:
            Larger radius for grouping raw points.
    """

    def __init__(
        self,
        lower_feature_channels,
        raw_feature_channels,
        lower_out_channels,
        raw_out_channels,
        sample_count,
        lower_neighbor_count,
        raw_neighbor_count,
        lower_radius,
        raw_radius,
    ):
        super().__init__()

        self.sample_count = sample_count

        self.lower_neighbor_count = lower_neighbor_count
        self.raw_neighbor_count = raw_neighbor_count

        self.lower_radius = lower_radius
        self.raw_radius = raw_radius

        # ---------------------------------------------------------
        # Branch 1:
        # PointNet over lower-level features.
        #
        # Input = relative XYZ + lower-level feature
        # ---------------------------------------------------------
        self.lower_mlp = SharedMLP(
            lower_feature_channels + 3,
            lower_out_channels,
        )

        # ---------------------------------------------------------
        # Branch 2:
        # PointNet over raw points.
        #
        # Input = relative XYZ + raw features
        # ---------------------------------------------------------
        self.raw_mlp = SharedMLP(
            raw_feature_channels + 3,
            raw_out_channels,
        )

        # ---------------------------------------------------------
        # Optional fusion after concatenation.
        #
        # This does NOT explicitly weight branches. It simply lets
        # later learned layers process the concatenated representation.
        # ---------------------------------------------------------
        self.output_channels = (
            lower_out_channels
            + raw_out_channels
        )

    def _ball_query(
        self,
        query_centers,
        support_points,
        radius,
        neighbor_count,
    ):
        """
        Radius-based grouping with nearest-neighbor fallback.

        Args:
            query_centers:
                (B, S, 3)

            support_points:
                (B, N, 3)

        Returns:
            indices:
                (B, S, K)
        """

        neighbor_count = min(
            neighbor_count,
            support_points.shape[1],
        )

        distances = torch.cdist(
            query_centers,
            support_points,
        )

        # Points outside the radius are masked.
        inside = distances <= radius

        masked_distances = distances.masked_fill(
            ~inside,
            float("inf"),
        )

        grouped_distances, grouped_indices = (
            masked_distances.topk(
                neighbor_count,
                dim=-1,
                largest=False,
            )
        )

        # ---------------------------------------------------------
        # Ball query can return fewer than K valid neighbors.
        # Use kNN only to fill missing slots.
        # ---------------------------------------------------------
        _, fallback_indices = distances.topk(
            neighbor_count,
            dim=-1,
            largest=False,
        )

        invalid = ~torch.isfinite(
            grouped_distances
        )

        grouped_indices = torch.where(
            invalid,
            fallback_indices,
            grouped_indices,
        )

        return grouped_indices

    def forward(
        self,
        raw_points,
        raw_features,
        lower_points,
        lower_features,
        return_branches=False,
    ):
        """
        Args:
            raw_points:
                (B, N, 3)

            raw_features:
                (B, N, C_raw)

            lower_points:
                (B, N_lower, 3)

            lower_features:
                (B, N_lower, C_lower)

        Returns:
            centers:
                (B, S, 3)

            output_features:
                (B, S, C_lower_out + C_raw_out)
        """

        # =========================================================
        # 1. Select current-level centers
        #
        # IMPORTANT:
        # Centers are selected from the lower-resolution point set.
        # This preserves the hierarchy:
        #
        # L0 → L1 → L2
        # =========================================================

        center_indices = farthest_point_sample(
            lower_points,
            self.sample_count,
        )

        centers = index_points(
            lower_points,
            center_indices,
        )

        # =========================================================
        # BRANCH 1
        # Lower-resolution features
        # =========================================================

        lower_neighbor_indices = self._ball_query(
            centers,
            lower_points,
            self.lower_radius,
            self.lower_neighbor_count,
        )

        lower_neighbor_points = index_points(
            lower_points,
            lower_neighbor_indices,
        )

        lower_neighbor_features = index_points(
            lower_features,
            lower_neighbor_indices,
        )

        lower_relative_points = (
            lower_neighbor_points
            - centers.unsqueeze(2)
        )

        lower_input = torch.cat(
            [
                lower_relative_points,
                lower_neighbor_features,
            ],
            dim=-1,
        )

        lower_local_features = self.lower_mlp(
            lower_input
        )

        lower_features_out = (
            lower_local_features
            .max(dim=2)
            .values
        )

        # =========================================================
        # BRANCH 2
        # Raw-point features
        # =========================================================

        raw_neighbor_indices = self._ball_query(
            centers,
            raw_points,
            self.raw_radius,
            self.raw_neighbor_count,
        )

        raw_neighbor_points = index_points(
            raw_points,
            raw_neighbor_indices,
        )

        raw_neighbor_features = index_points(
            raw_features,
            raw_neighbor_indices,
        )

        raw_relative_points = (
            raw_neighbor_points
            - centers.unsqueeze(2)
        )

        raw_input = torch.cat(
            [
                raw_relative_points,
                raw_neighbor_features,
            ],
            dim=-1,
        )

        raw_local_features = self.raw_mlp(
            raw_input
        )

        raw_features_out = (
            raw_local_features
            .max(dim=2)
            .values
        )

        # =========================================================
        # 3. Multi-resolution aggregation
        # =========================================================

        output_features = torch.cat(
            [
                lower_features_out,
                raw_features_out,
            ],
            dim=-1,
        )

        if return_branches:
            return (
                centers,
                output_features,
                lower_features_out,
                raw_features_out,
            )

        return centers, output_features


# ---------------------------------------------------------------------
# Full temporal PointNet++ feature extractor
# ---------------------------------------------------------------------

class PointNetPPFeatureExtractor(nn.Module):
    """Hierarchical PointNet++ feature extractor.

    Input:
        (B, T, P, C)

    Output:
        (B, T, feature_dim)

    When ``return_aux=True``, also returns the stage coordinates/features,
    point-propagated features, MRG branch outputs, and transform regularizer.
    """

    def __init__(
        self,
        in_channels=3,
        feature_dim=256,
        first_sample_count=256,
        second_sample_count=64,

        # New options
        use_input_transform=True,
        use_feature_transform=True,
        robust_grouping=True,
        radius_scale=2.0,
        first_sampling_mode="hybrid",
        second_sampling_mode="fps",

        # Multi-resolution grouping (L1 lower-level features + raw L0 points)
        # for the second set-abstraction level.
        use_mrg_second_stage=True,
        mrg_lower_neighbor_count=16,
        mrg_raw_neighbor_count=64,
        # NOTE: unlike the adaptive radius_scale used elsewhere in this file,
        # ball-query radii must be given in the same physical units as your
        # input xyz (e.g. meters). These defaults are a starting point --
        # tune them to your radar point cloud's actual spacing.
        mrg_lower_radius=0.5,
        mrg_raw_radius=1.0,

        # Optional segmentation branch
        num_segmentation_classes=None,
    ):
        super().__init__()

        if in_channels < 3:
            raise ValueError(
                "PointNet++ requires at least three XYZ channels"
            )

        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # -------------------------------------------------------------
        # Input transform: learned 3x3 alignment of XYZ.
        # -------------------------------------------------------------
        self.use_input_transform = use_input_transform

        if self.use_input_transform:
            self.input_transform = TransformNet(
                in_channels=3,
                transform_dim=3,
                hidden_channels=(32, 64, 128),
            )

        # -------------------------------------------------------------
        # First set abstraction.
        #
        # IMPORTANT:
        # There is no separate input projection anymore.
        #
        # The first SharedMLP sees:
        #
        #   relative XYZ + original non-XYZ point attributes
        #
        # -------------------------------------------------------------
        non_xyz_channels = in_channels - 3

        self.set_abstraction1 = PointNetSetAbstraction(
            in_channels=non_xyz_channels,
            out_channels=128,
            sample_count=first_sample_count,
            neighbor_count=32,
            sampling_mode=first_sampling_mode,
            robust_grouping=robust_grouping,
            radius_scale=radius_scale,
        )

        # -------------------------------------------------------------
        # Feature transform after first abstraction.
        # -------------------------------------------------------------
        self.use_feature_transform = use_feature_transform

        if self.use_feature_transform:
            self.feature_transform = TransformNet(
                in_channels=128,
                transform_dim=128,
                hidden_channels=(64, 128, 256),
            )

        # -------------------------------------------------------------
        # Second set abstraction (plain single-resolution variant).
        # -------------------------------------------------------------
        self.set_abstraction2 = PointNetSetAbstraction(
            in_channels=128,
            out_channels=feature_dim,
            sample_count=second_sample_count,
            neighbor_count=32,
            sampling_mode=second_sampling_mode,
            robust_grouping=robust_grouping,
            radius_scale=radius_scale,
        )

        # -------------------------------------------------------------
        # Second set abstraction (multi-resolution variant).
        #
        # Combines the already-abstracted L1 output (set_abstraction1)
        # with a wider-radius PointNet pass directly over the raw L0
        # points, so sparse/noisy regions still get a usable L2 feature.
        #
        # `use_mrg_second_stage` picks between this and set_abstraction2
        # in forward(); both modules exist so either can be selected
        # without changing the constructor signature.
        # -------------------------------------------------------------
        self.use_mrg_second_stage = use_mrg_second_stage

        lower_out_channels = feature_dim // 2
        raw_out_channels = feature_dim - lower_out_channels

        self.set_MRG = PointNetMRG(
            lower_feature_channels=128,
            raw_feature_channels=non_xyz_channels,
            lower_out_channels=lower_out_channels,
            raw_out_channels=raw_out_channels,
            sample_count=second_sample_count,
            lower_neighbor_count=mrg_lower_neighbor_count,
            raw_neighbor_count=mrg_raw_neighbor_count,
            lower_radius=mrg_lower_radius,
            raw_radius=mrg_raw_radius,
        )

        # -------------------------------------------------------------
        # Optional PointNet-style point segmentation branch.
        # -------------------------------------------------------------
        self.num_segmentation_classes = num_segmentation_classes

        if num_segmentation_classes is not None:
            self.segmentation_head = nn.Sequential(
                SharedMLP(
                    128 + in_channels,
                    [128, 64],
                ),
                nn.Linear(
                    64,
                    num_segmentation_classes,
                ),
            )

    def forward(self, points, return_aux=False):
        """
        Args:
            points:
                Tensor shaped (B, T, P, C)

            return_aux:
                If True, also return transform regularization and
                optional segmentation outputs.

        Returns:
            By default:
                frame_features: (B, T, feature_dim)

            If return_aux=True:
                frame_features, aux

        The auxiliary stage tensors use ``(B, T, ...)`` leading dimensions.
        """

        batch_size, temporal_window, point_count, channels = points.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} point channels, "
                f"got {channels}"
            )

        # -------------------------------------------------------------
        # Flatten temporal dimension.
        #
        # [B, T, P, C]
        #       ↓
        # [B*T, P, C]
        #
        # PointNet++ is applied independently to each frame.
        # -------------------------------------------------------------
        frame_points = points.reshape(
            batch_size * temporal_window,
            point_count,
            channels,
        )

        original_xyz = frame_points[..., :3]
        xyz = original_xyz

        # All non-XYZ radar features, e.g. intensity.
        point_attributes = frame_points[..., 3:]

        # -------------------------------------------------------------
        # 1. Input transform
        # -------------------------------------------------------------
        input_transform_matrix = None
        input_transform_loss = xyz.new_zeros(())

        if self.use_input_transform:
            input_transform_matrix = self.input_transform(
                xyz
            )

            xyz = torch.bmm(
                xyz,
                input_transform_matrix.transpose(1, 2),
            )

            input_transform_loss = transform_regularizer(
                input_transform_matrix
            )

        # -------------------------------------------------------------
        # 2. First PointNet++ set abstraction
        # -------------------------------------------------------------
        sampled_xyz, sampled_features, aux1 = (
            self.set_abstraction1(
                xyz,
                point_attributes,
            )
        )

        # -------------------------------------------------------------
        # 3. Feature transform
        # -------------------------------------------------------------
        feature_transform_matrix = None
        feature_transform_loss = sampled_features.new_zeros(())

        if self.use_feature_transform:
            feature_transform_matrix = self.feature_transform(
                sampled_features
            )

            sampled_features = torch.bmm(
                sampled_features,
                feature_transform_matrix.transpose(1, 2),
            )

            feature_transform_loss = transform_regularizer(
                feature_transform_matrix
            )

        # -------------------------------------------------------------
        # 4. Second PointNet++ set abstraction
        #
        # Either the multi-resolution (MRG) variant, which also pulls in
        # the raw L0 points, or the plain single-resolution variant.
        # -------------------------------------------------------------
        if self.use_mrg_second_stage:
            (
                second_xyz,
                global_region_features,
                mrg_lower_features,
                mrg_raw_features,
            ) = self.set_MRG(
                raw_points=xyz,
                raw_features=point_attributes,
                lower_points=sampled_xyz,
                lower_features=sampled_features,
                return_branches=True,
            )
            aux2 = {}
        else:
            second_xyz, global_region_features, aux2 = (
                self.set_abstraction2(
                    sampled_xyz,
                    sampled_features,
                )
            )

        # -------------------------------------------------------------
        # 5. Global aggregation over all regions
        #
        # [B*T, S2, feature_dim]
        #          ↓ max
        # [B*T, feature_dim]
        # -------------------------------------------------------------
        global_features = (
            global_region_features
            .max(dim=1)
            .values
        )

        global_features = global_features.view(
            batch_size,
            temporal_window,
            self.feature_dim,
        )

        if not return_aux:
            return global_features

        # -------------------------------------------------------------
        # Export intermediate representations for visualization and
        # self-supervised feature pretraining.
        # -------------------------------------------------------------
        aux = {
            "transform_regularization": (
                input_transform_loss
                + feature_transform_loss
            ),
            "input_transform": input_transform_matrix,
            "feature_transform": feature_transform_matrix,
            "first_random_ratio": aux1.get(
                "random_ratio",
                None,
            ),
            "second_random_ratio": aux2.get(
                "random_ratio",
                None,
            ),
        }

        if input_transform_matrix is not None:
            inverse_transform = torch.linalg.inv(input_transform_matrix)
            stage1_xyz = torch.bmm(
                sampled_xyz,
                inverse_transform.transpose(1, 2),
            )
            stage2_xyz = torch.bmm(
                second_xyz,
                inverse_transform.transpose(1, 2),
            )
        else:
            stage1_xyz = sampled_xyz
            stage2_xyz = second_xyz

        stage1_point_features = three_nn_interpolate(
            query_xyz=xyz,
            source_xyz=sampled_xyz,
            source_features=sampled_features,
        )
        stage2_point_features = three_nn_interpolate(
            query_xyz=xyz,
            source_xyz=second_xyz,
            source_features=global_region_features,
        )
        if self.use_mrg_second_stage:
            mrg_lower_point_features = three_nn_interpolate(
                query_xyz=xyz,
                source_xyz=second_xyz,
                source_features=mrg_lower_features,
            )
            mrg_raw_point_features = three_nn_interpolate(
                query_xyz=xyz,
                source_xyz=second_xyz,
                source_features=mrg_raw_features,
            )

        def temporal_shape(tensor):
            return tensor.reshape(
                batch_size,
                temporal_window,
                *tensor.shape[1:],
            )

        aux.update({
            "input_xyz": temporal_shape(original_xyz),
            "input_xyz_model": temporal_shape(xyz),
            "stage1_xyz": temporal_shape(stage1_xyz),
            "stage1_xyz_model": temporal_shape(sampled_xyz),
            "stage1_features": temporal_shape(sampled_features),
            "stage1_point_features": temporal_shape(stage1_point_features),
            "stage2_xyz": temporal_shape(stage2_xyz),
            "stage2_xyz_model": temporal_shape(second_xyz),
            "stage2_features": temporal_shape(global_region_features),
            "stage2_point_features": temporal_shape(stage2_point_features),
            "mrg_lower_features": (
                temporal_shape(mrg_lower_features)
                if self.use_mrg_second_stage
                else None
            ),
            "mrg_raw_features": (
                temporal_shape(mrg_raw_features)
                if self.use_mrg_second_stage
                else None
            ),
            "mrg_lower_point_features": (
                temporal_shape(mrg_lower_point_features)
                if self.use_mrg_second_stage
                else None
            ),
            "mrg_raw_point_features": (
                temporal_shape(mrg_raw_point_features)
                if self.use_mrg_second_stage
                else None
            ),
        })

        if self.num_segmentation_classes is not None:
            interpolated_features = three_nn_interpolate(
                query_xyz=xyz,
                source_xyz=sampled_xyz,
                source_features=sampled_features,
            )

            segmentation_input = torch.cat(
                [
                    xyz,
                    point_attributes,
                    interpolated_features,
                ],
                dim=-1,
            )

            segmentation_logits = self.segmentation_head(
                segmentation_input
            )

            segmentation_logits = segmentation_logits.view(
                batch_size,
                temporal_window,
                point_count,
                self.num_segmentation_classes,
            )

            aux["segmentation_logits"] = segmentation_logits

        return global_features, aux