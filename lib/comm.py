import rich
from rich.console import Console
import torch.distributed as dist
from decorator import decorator
from tensorboardX import SummaryWriter
from lib.common import gState
import warnings

__all__ = [
    "get_world_size", "get_rank", "is_main_process", "get_local_rank",
    "synchronize", "all_gather", "reduce_tensor", "get_device"
]


def get_world_size():
    if gState.gWorldSize > 0:
        return gState.gWorldSize
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    gState.gWorldSize = dist.get_world_size()
    return gState.gWorldSize


def reduce_tensor(tensor):
    """reduce tensor: sum(rt)/world_size"""
    if get_world_size() == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    elif isinstance(t, int):
        return float(t)
    elif isinstance(t, float):
        return t
    return t[0]


def get_rank():
    if gState.gRank >= 0:
        return gState.gRank
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    gState.gRank = dist.get_rank()
    return gState.gRank


def is_main_process():
    return get_rank() == 0


def synchronize():
    if not dist.is_available():
        warnings.warn("torch.distributed is not available.")
        return
    if not dist.is_initialized():
        warnings.warn("torch.distributed is not initialized.")
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data):
    obj_list = [None] * get_world_size()
    dist.all_gather_object(obj_list, data)
    return obj_list


class DistClassWrapper(type):
    """wrap functions that can only run in main process"""
    def __new__(cls, name, bases, attrs):
        @decorator
        def _decorater(f, *args, **kwargs):
            all_log = kwargs.pop("all", False)
            if all_log:
                return f(*args, **kwargs)
            if is_main_process():
                return f(*args, **kwargs)

        sub_cls = super().__new__(cls, name, bases, attrs)
        if "master_only" in attrs:
            for f in attrs["master_only"]:
                setattr(sub_cls, f, _decorater(getattr(sub_cls, f)))
        return sub_cls


class DistSummaryWriter(SummaryWriter, metaclass=DistClassWrapper):
    master_only = ["__init__", "add_scalar", "add_figure",
                   "add_graph", "add_histogram", "add_image", "add_text", "close"]


def get_local_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    if hasattr(gState, "local_rank"):
        return gState.local_rank
    assert False, "local_rank is not set"


def get_device():
    # TODO cpu
    return f"cuda:{get_local_rank()}"


class DistConsole(Console,
                  metaclass=DistClassWrapper):
    master_only = ["print"]


rich._console = DistConsole()
