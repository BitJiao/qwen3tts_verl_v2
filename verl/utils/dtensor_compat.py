"""Compatibility imports for DTensor APIs across PyTorch releases."""

try:
    from torch.distributed import DeviceMesh
except ImportError:
    from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.distributed.tensor import DTensor, Placement, Replicate, Shard, distribute_tensor
except ImportError:
    from torch.distributed._tensor import DTensor, Placement, Replicate, Shard, distribute_tensor

try:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
except Exception:
    from torch.distributed._tensor._utils import DTensorSpec

try:
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
except Exception:
    from torch.distributed._tensor._utils import compute_local_shape_and_global_offset

__all__ = [
    "DeviceMesh",
    "DTensor",
    "DTensorSpec",
    "Placement",
    "Replicate",
    "Shard",
    "compute_local_shape_and_global_offset",
    "distribute_tensor",
]
