import torch
from torch import nn
from torch.distributions.normal import Normal
from lib.common import gState


class AdaptiveShapingModule(nn.Module):
    def __init__(self, axis=0, sp_arg=10., mean_loss=True, learnable=None):
        super().__init__()
        if learnable is None:
            learnable = not gState.args.sp_fix
        if learnable:
            self.mean_arg = nn.parameter.Parameter(torch.tensor(0.))
            self.sp_arg = nn.parameter.Parameter(torch.tensor(sp_arg).log())
        else:
            mean_arg = torch.tensor(0.)
            self.register_buffer("mean_arg", mean_arg, persistent=False)
            sp_arg = torch.tensor(sp_arg).log()
            self.register_buffer("sp_arg", sp_arg, persistent=False)
        self.dim = axis
        self.mean_loss = mean_loss

    def forward(self, x: torch.Tensor):
        if not self.training:
            return torch.tensor(0., device=x.device)
        z = x.sort(dim=self.dim).values
        cdf_v = Normal(self.mean_arg, torch.exp(self.sp_arg)).cdf(z)
        N = z.shape[self.dim]
        tgt = torch.arange(1, 1+N)/(1+N)
        tgt = tgt.view(-1, 1, 1, 1)
        if self.mean_loss:
            return (cdf_v - tgt.cuda()).square().mean()
        else:
            return (cdf_v - tgt.cuda()).square().sum()
