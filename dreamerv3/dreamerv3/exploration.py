import jax.numpy as jnp
import ninjax as nj
import embodied.jax.nets as nn


class DisagreementEnsemble(nj.Module):

  act: str = 'silu'
  norm: str = 'rms'
  winit: str = 'trunc_normal_in'

  def __init__(self, output_dim, K=5, layers=2, units=128):
    self.output_dim = output_dim
    self.K = K
    self.layers = layers
    self.units = units
    self.kw = dict(winit=self.winit)

  def __call__(self, feat, action):
    x = jnp.concatenate([feat, action], -1)
    shape = x.shape[:-1]
    x = x.astype(nn.COMPUTE_DTYPE)
    x = x.reshape([-1, x.shape[-1]])
    preds = []
    for k in range(self.K):
      h = x
      for i in range(self.layers):
        h = self.sub(f'm{k}_l{i}', nn.Linear, self.units, **self.kw)(h)
        h = self.sub(f'm{k}_n{i}', nn.Norm, self.norm)(h)
        h = nn.act(self.act)(h)
      h = self.sub(f'm{k}_out', nn.Linear, self.output_dim, **self.kw)(h)
      preds.append(h)
    preds = jnp.stack(preds, axis=-2)  # (flat_batch, K, output_dim)
    preds = preds.reshape((*shape, self.K, self.output_dim))
    return preds


def disagreement(preds, eps=1e-8):
  var = jnp.var(preds, axis=-2)  # (..., output_dim)
  return jnp.sqrt(jnp.sum(var ** 2, axis=-1) + eps)  # (...)


def information_gain(preds, eps=1e-6):
  std = jnp.std(preds, axis=-2)  # (..., output_dim)
  return jnp.sum(jnp.log(std + eps), axis=-1)  # (...)
