"""Small deterministic shape-contract inputs, not a real KNN data pipeline."""

import torch


def build_hierarchy(config, batch_size=1, input_points=256, k=16, device="cpu"):
    sizes = [int(input_points)]
    for ratio in config.sub_sampling_ratio:
        sizes.append(max(1, sizes[-1] // int(ratio)))
    generator = torch.Generator(device="cpu").manual_seed(20260620)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    def randint(high, shape):
        return torch.randint(high, shape, generator=generator, dtype=torch.long).to(device)

    xyz, neigh_idx, sub_idx, interp_idx = [], [], [], []
    current_xyz = randn(batch_size, sizes[0], 3)
    for layer in range(config.num_layers):
        fine, coarse = sizes[layer:layer + 2]
        xyz.append(current_xyz)
        # Repeated neighbors allow tiny synthetic coarse levels to retain K=16.
        neigh_idx.append(randint(fine, (batch_size, fine, k)))
        sub_idx.append(randint(fine, (batch_size, coarse, k)))
        interp_idx.append(randint(coarse, (batch_size, fine, 1)))
        current_xyz = current_xyz[:, :coarse].contiguous()
    return {"xyz": xyz, "neigh_idx": neigh_idx, "sub_idx": sub_idx,
            "interp_idx": interp_idx,
            "features": randn(batch_size, input_points, config.input_channels)}
