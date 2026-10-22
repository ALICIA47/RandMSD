"""Strict loading of raw PyTorch weights and historical training archives."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch


def load_checkpoint(model, path, *, trusted=False, expected_sha256=None):
    """Load weights strictly on CPU, returning non-weight checkpoint metadata.

    Accepted inputs are a raw ``state_dict`` or an archive containing
    ``model_state_dict``. No prefix removal, missing-key tolerance, or shape
    adaptation is performed. Keys and shapes are checked before model mutation.

    ``weights_only=True`` is the default. If an old archive contains objects
    rejected by PyTorch's restricted loader, loading fails instead of silently
    enabling pickle. Pass ``trusted=True`` *only* for an independently trusted
    archive: unrestricted pickle can execute code. A hash verifies identity,
    not the safety of an otherwise untrusted archive.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise ValueError("Checkpoint SHA-256 does not match expected_sha256")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=not trusted)
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a state_dict or an archive mapping")
    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
        metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    else:
        state = payload
        metadata = {}
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint does not contain a nonempty model state_dict")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise ValueError("Every state_dict entry must be a string-keyed tensor")
    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    if missing or unexpected:
        raise RuntimeError("Strict checkpoint keys differ: missing={}, unexpected={}".format(
            missing, unexpected))
    mismatched = [key for key in expected if expected[key].shape != state[key].shape]
    if mismatched:
        raise RuntimeError("Strict checkpoint tensor shapes differ: {}".format(mismatched))
    model.load_state_dict(state, strict=True)
    return {"path": str(checkpoint_path.resolve()), "sha256": actual_sha256,
            "metadata": metadata}
