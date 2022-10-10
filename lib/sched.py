import math
from rich import print
from lib.common import TrainArg, gState


class BaseLR:
    @staticmethod
    def linear_warmup(self):
        progress = self.iters / self.warmup_iters
        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.init_lr + (lr - self.init_lr) * progress
        if self.iters == self.warmup_iters:
            self.iters = 0
            self.warmup = None

    def __init__(self, optimizer, T_max, eta_min=0, warmup=None, warmup_iters=None, init_lr=0.):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.init_lr = init_lr

        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        print(self.base_lr)
        print((len(group["params"]) for group in optimizer.param_groups))

    def state_dict(self):
        return {
            "iters": self.iters,
            "base_lr": self.base_lr,
            "warmup": self.warmup,
            "warmup_iters": self.warmup_iters,
        }

    def load_state_dict(self, state):
        self.iters = state["iters"]
        self.base_lr = state["base_lr"]
        self.warmup = state["warmup"]
        self.warmup_iters = state["warmup_iters"]

    def step(self, external_iter=None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        self.real_step()

    def real_step(self):
        raise NotImplementedError


class EmptyLR(BaseLR):
    def real_step(self):
        pass


class CosineAnnealingLR(BaseLR):
    def real_step(self):
        if self.warmup == 'linear' and self.iters <= self.warmup_iters:
            return self.linear_warmup(self)

        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * \
                (1 + math.cos(math.pi * self.iters / self.T_max)) / 2


class StepLR(BaseLR):

    def __init__(self, optimizer, T_max, eta_min=0, warmup=None, warmup_iters=None, init_lr=0,
                 milestones=None, step_beta=0.1):
        super().__init__(optimizer, T_max, eta_min, warmup, warmup_iters, init_lr)
        self.milestones = {int(T_max*milestone): j+1
                           for j, milestone in enumerate(milestones)}
        self.beta = step_beta

    def real_step(self):
        if self.warmup == 'linear' and self.iters <= self.warmup_iters:
            return self.linear_warmup(self)

        if self.iters in self.milestones:
            scale = self.beta ** self.milestones[self.iters]
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * scale
                print("[blue]Update LR {}".format(group['lr']))


def create_scheduler(optimizer, args) -> BaseLR:
    args: TrainArg = args
    len_epoch = gState.train_loader_len
    T_max = int(95 * len_epoch * args.epochs / 100)
    warmup_iters = int(5 * len_epoch * args.epochs / 100)
    gState.total_steps = T_max + warmup_iters
    print("[red]{} {} {}".format(len_epoch, T_max, warmup_iters))

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max,
                                      warmup='linear',
                                      warmup_iters=warmup_iters,
                                      init_lr=args.lr_init,
                                      eta_min=args.lr_end,)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer,
                           len_epoch * args.epochs,
                           init_lr=args.lr_init,
                           milestones=args.miles)
    else:
        raise ValueError("Unsupported scheduler.")
    if args.resume:
        scheduler.load_state_dict(gState.checkpoint["scheduler"])
        print(scheduler.state_dict())
    return scheduler
