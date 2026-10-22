"""RandMSD's frozen full-model API.

The public names are aliases, not wrappers: historical state-dict keys and
parameter initialization remain unchanged. ``ReferenceRandMSD`` is the direct
mathematical reference; ``RandMSD`` additionally enables the accepted memory
optimizations. Both consume a precomputed random-sampling hierarchy.
"""

from rsnative_corelite import CoreLiteConfig as ModelConfig
from rsnative_corelite import RSNativeCoreLiteOptimized as RandMSD
from rsnative_corelite import RSNativeCoreLiteV1 as ReferenceRandMSD

from .checkpoints import load_checkpoint

__all__ = ["RandMSD", "ReferenceRandMSD", "ModelConfig", "load_checkpoint"]
__version__ = "0.1.0"
