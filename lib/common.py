import dataclasses
from dataclasses import field
from typing import Any
import torch
from typing import List
import datetime
from lib.utils import ARG

time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


@dataclasses.dataclass
class TrainArg:
    data: str = ARG("").required()
    arch: str = ARG("none").short("a")
    workers: int = ARG(default=4).short("j")
    epochs: int = 100
    start_epoch: int = 0 # resume from unfinished training phase
    batch_size: int = ARG(256).short("b")
    lr: float = 0.1
    # ftlr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    rho_decay: float = 1e-4
    print_freq: int = 50
    resume: str = ""
    evaluate: bool = ARG(False).short("e")
    evaluate_model: str = "ignore"
    pretrained: bool = False
    clip_grad: bool = False
    dali_cpu: bool = False
    prof: int = -1
    seed: int = 0
    local_rank: int = 0
    sync_bn: bool = False
    # adjust_bn: bool = False
    opt_level: str = None
    disable_apex: bool = False
    keep_batchnorm_fp32: str = None
    loss_scale: str = "128.0"
    channels_last: bool = False
    work_dir: str = ARG(None).required()
    note: str = ""
    # nogit: bool = True
    # archive: str = None
    from_: str = "__main__"
    import_: str = ""
    num_classes: int = ARG().choices(1000, 10, 100)
    scheduler: str = ARG().choices("cosine", "step")
    miles: List[float] = ARG().sequence()
    lr_init: float = 0.
    lr_end: float = 0.
    one_batch: bool = False
    cifar: bool = False
    lmdb: bool = False
    lmdb_path: str = ""

    specify_lr: float = None
    sam: int = ARG().choices(0, 1, 2)

    mu_init: float = 0.
    rho_init: float = -3.
    extra_params: bool = False

    pri_mu: float = 0.
    pri_sigma: float = 1.

    kl_scale: float = 1.
    kl_type: str = ARG().choices("nokla", "cyc", "mono")
    aa: bool = False
    # option: int = 0
    # lars: bool = False
    # sig: float = 1.

    # model: str = ""
    finetune: bool = False
    find_unused_params: bool = False

    sscale: float = 1.
    stdscale: float = 1e-6
    blocks: List[int] = ARG().sequence(3, 4, 6, 3, size=4)
    skip_layers: List[int] = ARG().sequence()
    full_skip: List[int] = ARG().sequence()

    sp_pos: int = 0 # position of batch shaping
    # sp_name: str = ARG().choices("adaptive", "normal", "weibull", "lognormal")
    sp_arg: float = 10.0 # initial var
    sp_mean: bool = False # initial mean
    sp_fix: bool = False # make sp_arg and sp_mean not learnable
    sp_log_normal: bool = False
    sigscale: float = 1.0
    clamp: float = 1.0

    total_batch_size: int = ARG(-1).dummy()
    train_data: str = ARG("").dummy()
    val_data: str = ARG("").dummy()


@dataclasses.dataclass
class GlobalState:
    checkpoint: dict = None
    args: TrainArg = None
    custom_scalars: dict = field(default_factory=dict)
    custom_scalars_call: list = field(default_factory=list)
    all_label: list = field(default_factory=list)

    train_loader_len: int = 0
    val_loader_len: int = 0
    global_step: int = -1
    total_steps: int = 0
    gWorldSize: int = -1
    gRank: int = -1
    local_rank: int = -1
    summary: Any = None
    global_epoch: int = -1
    external_loss: torch.Tensor = 0
    sam: int = 0


gState = GlobalState()
