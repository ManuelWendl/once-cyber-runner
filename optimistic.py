import torch
import torch.nn as nn


class DisagreementEnsemble(nn.Module):
    """K parallel MLPs predicting the next-step encoder embedding from (h, z, a)."""

    def __init__(self, input_dim: int, out_dim: int, K: int = 5,
                 layers: int = 2, units: int = 512, act: str = "SiLU"):
        super().__init__()
        self.K = K
        self.out_dim = out_dim
        Act = getattr(nn, act)
        self.members = nn.ModuleList()
        for _ in range(K):
            layer_list = []
            in_d = input_dim
            for _ in range(layers):
                layer_list += [
                    nn.Linear(in_d, units, bias=True),
                    nn.RMSNorm(units, eps=1e-4, dtype=torch.float32),
                    Act(),
                ]
                in_d = units
            layer_list.append(nn.Linear(in_d, out_dim))
            self.members.append(nn.Sequential(*layer_list))

    def forward(self, h, z, a):
        z_flat = z.reshape(*z.shape[:-2], -1)
        x = torch.cat([h, z_flat, a], dim=-1)
        preds = torch.stack([m(x) for m in self.members], dim=-2)
        return preds


def disagreement(preds: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """preds: (..., K, D) -> (...,) L2 norm of per-dim variance across K."""
    var = preds.float().var(dim=-2)
    return torch.sqrt((var * var).sum(-1) + eps)
