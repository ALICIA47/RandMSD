"""Frozen RSNative-CoreLite-v1 segmentation backbone.

This is the correctness reference for publication, checkpoint loading, and
future kernel/memory optimization.  Do not optimize this file in place.  New
implementations belong in :mod:`rsnative_corelite.optimized` and must pass the
equivalence contracts in ``tests/``.

The data pipeline supplies an ``end_points`` dictionary containing:

* ``features``: ``[B, N, Cin]`` point attributes;
* ``xyz[l]``: ``[B, N_l, 3]`` point coordinates;
* ``neigh_idx[l]``: ``[B, N_l, K]`` same-level KNN indices;
* ``sub_idx[l]``: ``[B, N_(l+1), K]`` support indices around random centers;
* ``interp_idx[l]``: ``[B, N_l, 1]`` nearest coarse-center assignments.

The module names and seeded initialization match the frozen experiment graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1d(in_channels, out_channels, activation=True):
    layers = [
        nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.1),
    ]
    if activation:
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    return nn.Sequential(*layers)


def _conv2d(in_channels, out_channels, activation=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
    ]
    if activation:
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    return nn.Sequential(*layers)


def gather_points(values, indices):
    """Gather ``[B,N,C]`` values using ``[B,M,K]`` point indices."""
    if values.ndim != 3 or indices.ndim != 3:
        raise ValueError("gather_points expects values [B,N,C], indices [B,M,K]")
    batch_size, _, channels = values.shape
    flat_indices = indices.reshape(batch_size, -1).long()
    gathered = torch.gather(
        values,
        1,
        flat_indices.unsqueeze(-1).expand(-1, -1, channels),
    )
    return gathered.reshape(
        batch_size, indices.shape[1], indices.shape[2], channels
    )


def pool_random_support(feature, pool_idx):
    """Max-pool KNN supports around ordinary randomly retained centers."""
    gathered = gather_points(feature.transpose(1, 2).contiguous(), pool_idx)
    return gathered.amax(dim=2).transpose(1, 2).contiguous()


def nearest_interpolate(feature, interp_idx):
    """Interpolate ``[B,C,M]`` coarse features to ``[B,C,N]`` fine points."""
    if feature.ndim != 3 or interp_idx.ndim != 3:
        raise ValueError(
            "nearest_interpolate expects feature [B,C,M], indices [B,N,1]"
        )
    assignment = interp_idx.squeeze(-1).long()
    return torch.gather(
        feature,
        2,
        assignment.unsqueeze(1).expand(-1, feature.shape[1], -1),
    )


class NormalizedLocalMixer(nn.Module):
    """Local-mean-normalized geometry-semantic residual mixer."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        geo_channels = max(8, out_channels // 4)
        self.input_projection = _conv1d(in_channels, out_channels)
        self.geometry_projection = _conv2d(4, geo_channels)
        self.edge_projection = _conv2d(
            out_channels + geo_channels, out_channels
        )
        self.pool_fusion = _conv1d(2 * out_channels, out_channels, False)

    def forward(self, feature, xyz, neigh_idx):
        projected = self.input_projection(feature)
        neighbours = gather_points(
            projected.transpose(1, 2).contiguous(), neigh_idx
        ).permute(0, 3, 1, 2).contiguous()
        center = projected.unsqueeze(-1)

        neighbour_xyz = gather_points(xyz, neigh_idx)
        relative_xyz = neighbour_xyz - xyz.unsqueeze(2)
        relative_distance = torch.linalg.vector_norm(
            relative_xyz, dim=-1, keepdim=True
        )
        local_radius = relative_distance.mean(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        normalized_geometry = torch.cat(
            [
                relative_xyz / local_radius,
                relative_distance / local_radius,
            ],
            dim=-1,
        ).permute(0, 3, 1, 2).contiguous()

        geometry = self.geometry_projection(normalized_geometry)
        semantic_delta = neighbours - center
        edge = self.edge_projection(
            torch.cat([semantic_delta, geometry], dim=1)
        )
        max_response = edge.amax(dim=3)
        mean_response = edge.mean(dim=3)
        update = self.pool_fusion(
            torch.cat([max_response, mean_response], dim=1)
        )
        return F.leaky_relu(
            projected + update, negative_slope=0.2, inplace=True
        )


class RandomPoolTransition(nn.Module):
    """Parameter-free encoder transition at one random-sampling level."""

    def forward(
        self,
        fine_feature,
        fine_xyz,
        pool_idx,
        interp_idx,
        inject_detail=False,
        compute_detail=False,
        compute_alignment=False,
    ):
        del fine_xyz, interp_idx
        if inject_detail or compute_detail or compute_alignment:
            raise ValueError("CoreLite cannot compute or inject detail codes")
        return pool_random_support(fine_feature, pool_idx), None


class CoreContextBottleneck(nn.Module):
    """Pointwise residual bottleneck without a global-context branch."""

    def __init__(self, channels, match_experiment_initialization=True):
        super().__init__()
        self.local = nn.Sequential(
            _conv1d(channels, channels),
            _conv1d(channels, channels, False),
        )
        if match_experiment_initialization:
            # Historical experiments initialized a discarded context branch
            # before pruning.  Consume the same RNG without registering it.
            hidden = max(16, channels // 4)
            _ = nn.Sequential(
                nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            )

    def forward(self, feature):
        return F.leaky_relu(
            feature + self.local(feature),
            negative_slope=0.2,
            inplace=True,
        )


class CoreSynthesisTransition(nn.Module):
    """Nearest interpolation, additive skip, and residual refinement."""

    def __init__(self, coarse_channels, fine_channels):
        super().__init__()
        self.coarse_projection = _conv1d(
            coarse_channels, fine_channels, False
        )
        self.refinement = nn.Sequential(
            _conv1d(fine_channels, fine_channels),
            _conv1d(fine_channels, fine_channels, False),
        )

    def forward(self, coarse_feature, skip_feature, interp_idx):
        upsampled = nearest_interpolate(coarse_feature, interp_idx)
        upsampled = self.coarse_projection(upsampled)
        fused = skip_feature + upsampled
        return F.leaky_relu(
            fused + self.refinement(fused),
            negative_slope=0.2,
            inplace=True,
        )


def _consume_legacy_analysis_initialization(channels):
    """Consume RNG used by the pruned historical analysis branch."""
    _ = nn.Sequential(
        _conv1d(2 * channels + 2, channels),
        _conv1d(channels, channels, False),
    )


class RSNativeCoreLiteV1(nn.Module):
    """Frozen RSNative-CoreLite-v1 network used in the final experiments."""

    DEFAULT_WIDTHS = {
        "SemanticKITTI": (32, 64, 128, 256),
        "S3DIS": (32, 64, 128, 256, 384),
        "SensatUrban": (32, 64, 128, 256, 384),
        "Toronto3D": (32, 64, 128, 256, 384),
    }
    EXPECTED_LAYERS = {
        "SemanticKITTI": 4,
        "S3DIS": 5,
        "SensatUrban": 5,
        "Toronto3D": 5,
    }
    EXPECTED_CLASSES = {
        "SemanticKITTI": 19,
        "S3DIS": 13,
        "SensatUrban": 13,
        "Toronto3D": 8,
    }
    EXPECTED_RATIOS = {
        "SemanticKITTI": (4, 4, 4, 4),
        "S3DIS": (4, 4, 4, 4, 2),
        "SensatUrban": (4, 4, 4, 4, 2),
        "Toronto3D": (4, 4, 4, 4, 2),
    }
    EXPECTED_INPUT_CHANNELS = {
        "SemanticKITTI": 4,
        "S3DIS": 6,
        "SensatUrban": 6,
        "Toronto3D": 6,
    }

    def __init__(self, config, match_experiment_initialization=True):
        super().__init__()
        dataset_name = str(config.name)
        if dataset_name not in self.DEFAULT_WIDTHS:
            raise ValueError(
                "RandMSD supports S3DIS, SemanticKITTI, and SensatUrban"
            )

        num_layers = int(config.num_layers)
        if num_layers != self.EXPECTED_LAYERS[dataset_name]:
            raise ValueError(
                "{} requires {} hierarchy layers, got {}".format(
                    dataset_name,
                    self.EXPECTED_LAYERS[dataset_name],
                    num_layers,
                )
            )
        num_classes = int(config.num_classes)
        if num_classes != self.EXPECTED_CLASSES[dataset_name]:
            raise ValueError(
                "{} requires {} classes, got {}".format(
                    dataset_name,
                    self.EXPECTED_CLASSES[dataset_name],
                    num_classes,
                )
            )
        if int(getattr(config, "k_n", 16)) != 16:
            raise ValueError("RSNative-CoreLite-v1 is frozen with K=16")

        configured_ratios = tuple(
            getattr(
                config,
                "sub_sampling_ratio",
                self.EXPECTED_RATIOS[dataset_name],
            )
        )
        if configured_ratios != self.EXPECTED_RATIOS[dataset_name]:
            raise ValueError(
                "{} requires sampling ratios {}, got {}".format(
                    dataset_name,
                    self.EXPECTED_RATIOS[dataset_name],
                    configured_ratios,
                )
            )

        expected_widths = self.DEFAULT_WIDTHS[dataset_name]
        configured_widths = getattr(config, "rs_native_widths", None)
        widths = (
            list(expected_widths)
            if configured_widths is None
            else [int(value) for value in configured_widths]
        )
        if tuple(widths) != expected_widths:
            raise ValueError(
                "{} requires frozen widths {}, got {}".format(
                    dataset_name, expected_widths, tuple(widths)
                )
            )

        expected_input_channels = self.EXPECTED_INPUT_CHANNELS[dataset_name]
        if dataset_name == "SemanticKITTI":
            if not bool(getattr(config, "use_remission", False)):
                raise ValueError(
                    "Frozen SemanticKITTI CoreLite requires remission"
                )
            input_channels = 4
        else:
            input_channels = int(
                getattr(config, "input_channels", expected_input_channels)
            )
        if input_channels != expected_input_channels:
            raise ValueError(
                "{} requires {} input channels, got {}".format(
                    dataset_name, expected_input_channels, input_channels
                )
            )

        self.dataset_name = dataset_name
        self.widths = widths
        self.stem = nn.Sequential(
            _conv1d(input_channels, widths[0]),
            _conv1d(widths[0], widths[0]),
        )

        self.local_mixers = nn.ModuleList()
        self.analysis_transitions = nn.ModuleList()
        previous_width = widths[0]
        for width in widths:
            self.local_mixers.append(
                NormalizedLocalMixer(previous_width, width)
            )
            if match_experiment_initialization:
                _consume_legacy_analysis_initialization(width)
            self.analysis_transitions.append(RandomPoolTransition())
            previous_width = width

        self.bottleneck = CoreContextBottleneck(
            widths[-1],
            match_experiment_initialization=match_experiment_initialization,
        )
        self.synthesis_transitions = nn.ModuleList()
        coarse_width = widths[-1]
        for fine_width in reversed(widths):
            self.synthesis_transitions.append(
                CoreSynthesisTransition(coarse_width, fine_width)
            )
            coarse_width = fine_width

        head_width = max(32, widths[0])
        self.head = nn.Sequential(
            _conv1d(widths[0], 64),
            _conv1d(64, head_width),
        )
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Conv1d(
            head_width, num_classes, kernel_size=1, bias=True
        )

    def forward(self, end_points):
        feature = end_points["features"].transpose(1, 2).contiguous()
        feature = self.stem(feature)

        skips = []
        for layer_idx, (mixer, transition) in enumerate(
            zip(self.local_mixers, self.analysis_transitions)
        ):
            feature = mixer(
                feature,
                end_points["xyz"][layer_idx],
                end_points["neigh_idx"][layer_idx],
            )
            skips.append(feature)
            feature, _ = transition(
                feature,
                end_points["xyz"][layer_idx],
                end_points["sub_idx"][layer_idx],
                end_points["interp_idx"][layer_idx],
            )

        feature = self.bottleneck(feature)
        for decoder_idx, transition in enumerate(self.synthesis_transitions):
            encoder_idx = len(skips) - decoder_idx - 1
            feature = transition(
                feature,
                skips[encoder_idx],
                end_points["interp_idx"][encoder_idx],
            )

        point_feature = self.head(feature)
        end_points["point_features"] = point_feature.transpose(
            1, 2
        ).contiguous()
        end_points["logits"] = self.classifier(self.dropout(point_feature))
        return end_points

    def frozen_spec(self):
        """Return the publication-facing architectural contract."""
        return {
            "version": "RSNative-CoreLite-v1",
            "dataset": self.dataset_name,
            "widths": tuple(self.widths),
            "mixer_input": "geometry_semantic",
            "geometry_normalization": "local_mean",
            "mixer_pooling": "max_mean",
            "encoder_transition": "parameter_free_knn_max_pool",
            "global_context": False,
            "analysis_synthesis_detail": False,
            "decoder": "nearest_additive_skip_residual_refinement",
        }


RSNativeCoreLite = RSNativeCoreLiteV1
