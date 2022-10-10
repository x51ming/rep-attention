import random
import numpy as np
import torch
import time
from decorator import decorator
import argparse
from typing import List, Dict, Any
import dataclasses
from rich import print


@decorator
def timeit(func, *args, **kwargs):
    t0 = time.time()
    v = func(*args, **kwargs)
    t1 = time.time()
    print("call %r [%rs]" % (func, t1 - t0))
    return v


class _TypeDefault:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def __repr__(self) -> str:
        return '@%r' % self.value


def add_dataclass_arguments(self: argparse.ArgumentParser,
                            argument_class: object,
                            copy_to_scope=None):
    one_arg_type = {int, float, str, bool}
    nargs_type = {List[int], List[float], List[str]}
    supported_type = one_arg_type.union(nargs_type)

    argument_dataclass = dataclasses.dataclass(argument_class)
    for field in dataclasses.fields(argument_dataclass):
        kwargs: Dict[str, Any] = field.metadata.copy()
        if kwargs.get("dummy", False):
            continue
        # process type
        field_type = field.type
        assert field_type in supported_type
        assert_type = field_type
        prim_type = field_type
        # process list type
        if field_type in nargs_type:
            prim_type = field_type.__args__[0]
            assert_type = field_type.__origin__
            if 'nargs' not in kwargs and kwargs.get("action") != "append":
                kwargs['nargs'] = '+'
        if prim_type != bool:
            kwargs["type"] = prim_type

        # process default
        if isinstance(field.default, _TypeDefault):
            kwargs["default"] = field.default.get()
        elif isinstance(field.default, dataclasses._MISSING_TYPE):
            kwargs["default"] = None
        else:
            kwargs["default"] = field.default
        assert kwargs["default"] is None or \
            isinstance(kwargs["default"], assert_type),\
            f"{field.name} [{assert_type}] = {kwargs['default']}"
        # process field names
        field_name = f"--{field.name}"
        field_names = [field_name]
        if "_" in field_name:
            field_names.append(field_name.replace("_", "-"))
        if "short" in kwargs:
            field_names.append(kwargs.pop("short"))

        # process action
        if field_type is bool:
            if "action" in kwargs:
                print("[red]Overwrite action with `store_const` action")
            kwargs["action"] = "store_false" if kwargs["default"] == True else "store_true"

        self.add_argument(*field_names, **kwargs)


class ARG(dataclasses.Field):
    def __init__(self, default=dataclasses.MISSING,
                 *,
                 default_factory=dataclasses.MISSING,
                 init=True, repr=True, hash=None, compare=True, metadata=None):
        if default is not dataclasses.MISSING \
                and default_factory is not dataclasses.MISSING:
            raise ValueError(
                'cannot specify both default and default_factory')
        if default is dataclasses.MISSING:
            default = None
        super().__init__(default, default_factory, init, repr,
                         hash, compare, metadata)
        self.metadata = dict()

    def help(self, help: str):
        self.metadata["help"] = help
        return self

    def choices(self, *values):
        self.metadata["choices"] = values
        self.default = values[0]
        return self

    def sequence(self, *values, size: int = None,):
        self.default = _TypeDefault(list(values))
        if size is not None:
            self.metadata['nargs'] = size
        else:
            self.metadata['nargs'] = '+'
        return self

    def required(self):
        self.metadata["required"] = True
        return self

    def short(self, shortname):
        assert len(shortname) == 1
        self.metadata["short"] = "-" + shortname
        return self

    def append(self, *values):
        if len(values) == 0:
            self.default = None
        else:
            self.default = _TypeDefault(list(values))
        self.metadata["action"] = "append"
        return self

    def custom_action(self, action):
        self.metadata["action"] = action
        return self

    def dummy(self):
        self.metadata["dummy"] = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def any_import(from_: str, import_: str):
    import importlib as imp
    if "/" in from_:
        from_ = from_.replace("/", ".")
    from_ = from_.rstrip(".py")
    _module = imp.import_module(from_)
    print("[green]", _module)
    model_fn = getattr(_module, import_)
    print("[green]", model_fn)
    return model_fn
