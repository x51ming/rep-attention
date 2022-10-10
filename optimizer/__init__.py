from .sgd import SGD
from torch import nn


def model_params(model: nn.Module, args):
    with_wd = []
    rho_dec = []
    no_wd = []
    for n, p in model.named_parameters():
        if "sp_arg" in n or "mean_arg" in n:
            no_wd.append(p)
            continue
        if not "rho_" in n:
            with_wd.append(p)
        else:
            rho_dec.append(p)
    if rho_dec:
        return [
            {
                "params": with_wd,
            },
            {
                "params": rho_dec,
                "weight_decay": 0.,
                "rho_decay": (args.rho_decay, args.pri_sigma)
            },
            {
                "params": no_wd,
                "weight_decay": 0.
            }
        ]
    else:
        return with_wd
