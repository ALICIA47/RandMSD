"""Accepted implementation optimizations for frozen CoreLite-v1.

The reference module remains immutable. Evaluation uses exact center-axis
chunking to bound neighborhood activation memory. Training checkpoints the
first high-resolution local mixer and restores BatchNorm running buffers after
backward recomputation. Parameters, state-dict names, and optimizer updates are
unchanged.
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .reference import RSNativeCoreLiteV1, gather_points


def gather_channel_first(values, indices):
    """Gather ``[B,C,N]`` values directly into ``[B,C,M,K]``.

    Expanded indices are views.  Gathering on the point dimension avoids the
    reference transpose/contiguous/gather/permute/contiguous sequence without
    changing any selected feature value.
    """
    if values.ndim != 3 or indices.ndim != 3:
        raise ValueError(
            "gather_channel_first expects values [B,C,N], indices [B,M,K]"
        )
    if values.shape[0] != indices.shape[0]:
        raise ValueError("values and indices must have the same batch size")
    batch_size, channels, _ = values.shape
    num_centers, num_neighbors = indices.shape[1:]
    gather_idx = indices.reshape(batch_size, 1, -1).long().expand(
        batch_size, channels, num_centers * num_neighbors
    )
    gathered = torch.gather(values, 2, gather_idx)
    return gathered.reshape(
        batch_size, channels, num_centers, num_neighbors
    )


@contextlib.contextmanager
def preserve_batch_norm_running_stats(module):
    """Restore BatchNorm buffers after checkpoint backward recomputation."""
    snapshots = []
    for child in module.modules():
        if (
            isinstance(child, nn.modules.batchnorm._BatchNorm)
            and child.track_running_stats
        ):
            snapshots.append(
                (
                    child,
                    child.running_mean.detach().clone(),
                    child.running_var.detach().clone(),
                    child.num_batches_tracked.detach().clone(),
                )
            )
    try:
        yield
    finally:
        with torch.no_grad():
            for child, running_mean, running_var, batches in snapshots:
                child.running_mean.copy_(running_mean)
                child.running_var.copy_(running_var)
                child.num_batches_tracked.copy_(batches)


def _eval_mixer_chunk(mixer, projected, xyz, neigh_idx, start, end):
    """Compute one exact center-axis chunk of a local mixer update."""
    chunk_idx = neigh_idx[:, start:end]
    center_feature = projected[:, :, start:end]
    semantic_delta = gather_channel_first(projected, chunk_idx)
    semantic_delta.sub_(center_feature.unsqueeze(-1))

    neighbour_xyz = gather_points(xyz, chunk_idx)
    relative_xyz = neighbour_xyz - xyz[:, start:end].unsqueeze(2)
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

    geometry = mixer.geometry_projection(normalized_geometry)
    edge = mixer.edge_projection(
        torch.cat([semantic_delta, geometry], dim=1)
    )
    max_response = edge.amax(dim=3)
    mean_response = edge.mean(dim=3)
    return mixer.pool_fusion(
        torch.cat([max_response, mean_response], dim=1)
    )


def eval_mixer_forward(mixer, feature, xyz, neigh_idx, chunk_size=8192):
    """Reference mixer math with exact bounded-memory center chunking."""
    projected = mixer.input_projection(feature)
    if chunk_size > 0 and neigh_idx.shape[1] > chunk_size:
        updates = [
            _eval_mixer_chunk(
                mixer,
                projected,
                xyz,
                neigh_idx,
                start,
                min(start + chunk_size, neigh_idx.shape[1]),
            )
            for start in range(0, neigh_idx.shape[1], chunk_size)
        ]
        return F.leaky_relu(
            projected + torch.cat(updates, dim=2),
            negative_slope=0.2,
            inplace=True,
        )

    semantic_delta = gather_channel_first(projected, neigh_idx)
    semantic_delta.sub_(projected.unsqueeze(-1))

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

    geometry = mixer.geometry_projection(normalized_geometry)
    edge = mixer.edge_projection(
        torch.cat([semantic_delta, geometry], dim=1)
    )
    max_response = edge.amax(dim=3)
    mean_response = edge.mean(dim=3)
    update = mixer.pool_fusion(
        torch.cat([max_response, mean_response], dim=1)
    )
    return F.leaky_relu(
        projected + update, negative_slope=0.2, inplace=True
    )


class RSNativeCoreLiteOptimized(RSNativeCoreLiteV1):
    """Frozen CoreLite-v1 with accepted inference/training memory layouts."""

    def __init__(
        self,
        config,
        match_experiment_initialization=True,
        eval_chunk_size=8192,
        activation_checkpoint_layers=1,
    ):
        super().__init__(config, match_experiment_initialization)
        self.eval_chunk_size = int(eval_chunk_size)
        self.activation_checkpoint_layers = int(
            activation_checkpoint_layers
        )
        if self.eval_chunk_size < 0:
            raise ValueError("eval_chunk_size must be non-negative")
        if not 0 <= self.activation_checkpoint_layers <= len(
            self.local_mixers
        ):
            raise ValueError(
                "activation_checkpoint_layers must be between 0 and num_layers"
            )

    def forward(self, end_points):
        feature = end_points["features"].transpose(1, 2).contiguous()
        feature = self.stem(feature)

        skips = []
        for layer_idx, (mixer, transition) in enumerate(
            zip(self.local_mixers, self.analysis_transitions)
        ):
            if (
                self.training
                and layer_idx < self.activation_checkpoint_layers
            ):
                def mixer_forward(
                    current_feature,
                    current_xyz,
                    current_neigh_idx,
                    current_mixer=mixer,
                ):
                    return current_mixer(
                        current_feature,
                        current_xyz,
                        current_neigh_idx,
                    )

                def checkpoint_context(current_mixer=mixer):
                    return (
                        contextlib.nullcontext(),
                        preserve_batch_norm_running_stats(current_mixer),
                    )

                feature = checkpoint(
                    mixer_forward,
                    feature,
                    end_points["xyz"][layer_idx],
                    end_points["neigh_idx"][layer_idx],
                    use_reentrant=False,
                    context_fn=checkpoint_context,
                )
            elif self.training:
                feature = mixer(
                    feature,
                    end_points["xyz"][layer_idx],
                    end_points["neigh_idx"][layer_idx],
                )
            else:
                feature = eval_mixer_forward(
                    mixer,
                    feature,
                    end_points["xyz"][layer_idx],
                    end_points["neigh_idx"][layer_idx],
                    chunk_size=self.eval_chunk_size,
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
