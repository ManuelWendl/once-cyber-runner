import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import exploration as expl
from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    # `prior_*` keys are SOOPER Mode 2 replay slots — loss targets for the
    # distilled risk head and the V^pi~_r head, never encoded or
    # reconstructed (same treatment as `reward`). `prior_v` is the raw
    # survival-prior V_prior that the risk head regresses; `prior_risk` is
    # kept for analysis logging.
    exclude = ('is_first', 'is_last', 'is_terminal', 'reward',
               'prior_risk', 'prior_active', 'prior_v')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')
    ball_space = elements.Space(np.float32, (2,))
    self.ball_pos_head = embodied.jax.MLPHead(
        ball_space, **config.ball_pos_head, name='ball_pos')
    self.ball_vel_head = embodied.jax.MLPHead(
        ball_space, **config.ball_vel_head, name='ball_vel')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.exploration_reward = str(config.exploration.reward)
    self.exploration_lambd = float(config.exploration.lambd)
    self.exploration_task_reward = bool(config.exploration.task_reward)
    self.has_reward_head = (
        self.exploration_task_reward or self.exploration_reward == 'none')
    if self.has_reward_head:
      self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')

    # SOOPER Mode 2 heads — built only when sooper.mode2_enabled, so plain
    # OPAX / Mode-1-only runs are byte-identical to before.
    #   risk     : distills the gate's risk_critic ([0,1] scalar) onto the
    #              latent, so the in-imagination switching condition Φ≥d
    #              can be evaluated on imagined states.
    #   priorval : V^pi~_r — the explorer's intrinsic-reward value under the
    #              PRIOR policy; the terminal value the imagined return
    #              bootstraps from when the switch fires.
    self.mode2_enabled = bool(config.sooper.mode2_enabled)
    if self.mode2_enabled:
      assert self.exploration_reward != 'none', (
          'sooper.mode2_enabled requires an OPAX exploration reward — '
          'V^pi~_r is the value of the explorer\'s intrinsic reward.')
      self.risk = embodied.jax.MLPHead(scalar, **config.riskhead, name='risk')
      self.priorval = embodied.jax.MLPHead(
          scalar, **config.priorvalhead, name='priorval')
      # Polyak-averaged slow copy of priorval — used as the bootstrap target
      # in the priorval lambda-return so the head doesn't self-bootstrap.
      # v1 ran with boot=sg(pv) which is a self-reinforcing positive-feedback
      # loop: a too-high pv pulls the target even higher, head loss grew
      # 0.41 → 0.49 over 100k steps and pv ran to ~3-47 vs ~0.4-2 real
      # targets. Mirrors the slowval/val pattern (same slowvalue config).
      self.slowpriorval = embodied.jax.SlowModel(
          embodied.jax.MLPHead(
              scalar, **config.priorvalhead, name='slowpriorval'),
          source=self.priorval, **config.slowvalue)

    self.modules = [
        self.dyn, self.enc, self.dec, self.con,
        self.ball_pos_head, self.ball_vel_head, self.pol, self.val]
    if self.has_reward_head:
      self.modules.insert(3, self.rew)
    if self.mode2_enabled:
      self.modules += [self.risk, self.priorval, self.slowpriorval]

    if self.exploration_reward != 'none':
      rssm_cfg = config.dyn[config.dyn.typ]
      feat_dim = rssm_cfg.deter + rssm_cfg.stoch * rssm_cfg.classes
      act_dim = sum(s.shape[0] for s in act_space.values())
      token_dim = self._compute_token_dim(enc_space, config.enc[config.enc.typ])
      self.ensemble = expl.DisagreementEnsemble(
          token_dim, name='ensemble', **config.exploration.ensemble)
      self.modules.append(self.ensemble)

    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    if self.exploration_reward == 'none':
      scales.pop('ensemble', None)
    if not self.has_reward_head:
      scales.pop('rew', None)
    if not self.config.repval_loss:
      scales.pop('repval', None)
    if not self.mode2_enabled:
      scales.pop('risk', None)
      scales.pop('priorval', None)
    self.scales = scales

  @property
  def policy_keys(self):
    # `con` (continuation head) is included so SOOPER's mode='sooper' branch
    # in policy() can compute risk_horizon. The head is small (~2 MLP layers);
    # plain OPAX runs pay negligible extra memory.
    return '^(enc|dyn|dec|pol|con)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    # SOOPER: when mode=='sooper', additionally compute K-step risk_horizon
    # under the OPAX actor and expose it to the PolicySwitcher via outs.
    # Plain OPAX (mode='train' / 'eval') doesn't pay the imagination cost.
    if mode == 'sooper':
      K = 10  # imagination horizon — keep small so the JIT scan is cheap
      actor_fn = lambda c: sample(self.pol(self.feat2tensor(c), bdims=1))
      _, feat_imag, _ = self.dyn.imagine(
          dyn_carry, actor_fn, length=K, training=False)
      cont_probs = self.con(self.feat2tensor(feat_imag), bdims=2).prob(1)
      # Expose both cont-derived risks so the PolicySwitcher can pick one
      # at runtime (and log all of them simultaneously for comparison).
      out['risk_cont_product'] = 1.0 - jnp.prod(cont_probs, axis=1)  # (B,)
      out['risk_cont_max'] = jnp.max(1.0 - cont_probs, axis=1)        # (B,)
      # NOTE: σ_n(s_t, a_t) exposure intentionally omitted here. Calling
      # self.ensemble(...) during policy() triggers Ninjax's "create new
      # state outside init" guard during the train.py:228 warmup
      # (driver(policy, steps=10) runs BEFORE any train_step, so the
      # ensemble's lazily-created params don't exist yet). PolicySwitcher
      # falls back to σ=0 when 'sigma_disagreement' is absent from out,
      # which makes the λ_pessimism term identically zero — fine for the
      # default config. Re-introduce σ later by either (a) computing it
      # inside the imagination loop where the ensemble already runs, or
      # (b) eager-initialising ensemble params before warmup data
      # collection.
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    if self.mode2_enabled:
      self.slowpriorval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    if self.has_reward_head:
      inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
      losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    if self.mode2_enabled:
      # SOOPER Mode 2: distill V_prior (the survival prior's value head)
      # onto the latent so the in-imagination switching condition can
      # recover risk_critic via clip(1 - V/V_norm). Distilling V_prior
      # directly (continuous, range ~40-135) avoids the 97% zero-mass
      # imbalance that collapsed a risk_critic-target version of this head
      # to predicting ~0 everywhere. Stop-grad input — auxiliary readout,
      # must not perturb the working world-model representation.
      # obs['prior_v'] is what the PolicySwitcher wrote into replay (0.0
      # on no-gate steps, e.g. warmup — harmless).
      losses['risk'] = self.risk(
          sg(self.feat2tensor(repfeat)), 2).loss(obs['prior_v'])
      metrics['sooper/v_pred_mean']   = self.risk(
          sg(self.feat2tensor(repfeat)), 2).pred().mean()
      metrics['sooper/v_target_mean'] = obs['prior_v'].mean()
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))
    inp_phys = self.feat2tensor(repfeat)
    pos_target = obs['states'][:, 1:, 2:4]
    losses['ball_pos'] = jnp.pad(
        self.ball_pos_head(inp_phys[:, :-1], 2).loss(pos_target),
        ((0, 0), (0, 1)))
    vel_target = pos_target - obs['states'][:, :-1, 2:4]
    losses['ball_vel'] = jnp.pad(
        self.ball_vel_head(inp_phys[:, :-1], 2).loss(vel_target),
        ((0, 0), (0, 1)))

    # Ensemble training: predict next encoder tokens from current state + action
    if self.exploration_reward != 'none':
      feat_t = sg(self.feat2tensor(repfeat))[:, :-1]
      act_t = jnp.concatenate(
          [prevact[k][:, 1:] for k in sorted(prevact)], -1)
      tokens_target = sg(tokens[:, 1:])
      preds = self.ensemble(feat_t, act_t)  # (B, T-1, K, out_dim)
      ens_loss = ((preds - tokens_target[..., None, :]) ** 2).mean((-1, -2))
      losses['ensemble'] = jnp.pad(ens_loss, ((0, 0), (0, 1)))
      metrics['exploration/ensemble_loss'] = ens_loss.mean()
      ens_var = jnp.var(preds, axis=-2)  # (B, T-1, output_dim)
      metrics['exploration/ensemble_variance'] = ens_var.mean()

      if self.mode2_enabled:
        # SOOPER Mode 2: V^pi~_r — value of the explorer's intrinsic reward
        # under the PRIOR policy. Trained only on prior-driven transitions
        # (obs['prior_active']==1): there the realized trajectory IS a prior
        # rollout, so its intrinsic-reward lambda-return is an unbiased
        # V^pi~_r target. Reuses `preds` (real-trajectory ensemble preds)
        # from just above; intrinsic reward matches the imagination bonus
        # (exploration_lambd * disagreement). Bootstrap from slowpriorval
        # (Polyak-averaged copy of priorval) NOT from pv itself — v1
        # self-bootstrapped via boot=sg(pv) which created a positive feedback
        # loop, runaway pv (3-47) and growing loss (0.41 → 0.49).
        intr = self.exploration_lambd * expl.disagreement(preds)  # (B,T-1)
        intr = jnp.pad(intr, ((0, 0), (0, 1)))                    # (B,T)
        pv_head = self.priorval(sg(self.feat2tensor(repfeat)), 2)
        pv = pv_head.pred()                                       # (B,T)
        pv_slow = sg(self.slowpriorval(
            sg(self.feat2tensor(repfeat)), 2).pred())             # (B,T)
        disc = 1 - 1 / self.config.horizon
        pv_ret = lambda_return(
            f32(obs['is_last']), f32(obs['is_terminal']), intr,
            pv_slow, pv_slow, disc, 0.95)                          # (B,T-1)
        pv_ret = jnp.concatenate([pv_ret, 0 * pv_ret[:, -1:]], 1)  # (B,T)
        w = obs['prior_active']                                   # (B,T)
        losses['priorval'] = jnp.pad(
            (w * pv_head.loss(sg(pv_ret)))[:, :-1], ((0, 0), (0, 1)))
        metrics['sooper/priorval_mean'] = pv.mean()
        metrics['sooper/priorval_slow_mean'] = pv_slow.mean()
        metrics['sooper/intr_reward_mean'] = intr.mean()
        metrics['sooper/prior_active_frac'] = w.mean()

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    if self.has_reward_head:
      imag_rew = self.rew(inp, 2).pred()

    # Exploration bonus in imagination
    if self.exploration_reward != 'none':
      img_act_concat = jnp.concatenate(
          [imgact[k] for k in sorted(imgact)], -1)
      ens_preds = sg(self.ensemble(sg(inp), img_act_concat))
      if self.exploration_reward == 'disagreement':
        bonus = expl.disagreement(ens_preds)
      else:
        bonus = expl.information_gain(ens_preds)
      bonus = self.exploration_lambd * bonus
      metrics['exploration/bonus_mean'] = bonus.mean()
      metrics['exploration/bonus_std'] = bonus.std()
      if self.exploration_task_reward:
        imag_rew = imag_rew + bonus
      else:
        imag_rew = bonus

    # SOOPER Mode 2: evaluate the distilled V_prior head + V^pi~_r head on
    # imagined latents so imag_loss can apply the in-imagination switching
    # condition. The risk head outputs V_prior (not risk_critic); imag_loss
    # recovers risk via the same clip(1-V/V_norm) the gate uses. sg() —
    # these heads are trained by their own losses above, not through the
    # actor loss. Reuses the v10-locked Mode 1 calibration.
    mode2_kw = {}
    if self.mode2_enabled:
      mode2_kw = dict(
          mode2=True,
          v_imag=sg(self.risk(inp, 2).pred()),
          priorval_imag=sg(self.priorval(inp, 2).pred()),
          budget_d=float(self.config.sooper.budget_d),
          gamma_cost=float(self.config.sooper.gamma_cost),
          V_norm=float(self.config.sooper.V_norm),
      )
    los, imgloss_out, mets = imag_loss(
        imgact,
        imag_rew,
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **mode2_kw,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)
      if video.shape[-1] == 1:
        video = jnp.repeat(video, 3, axis=-1)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    # Ensemble disagreement heatmap over the report batch
    if self.exploration_reward != 'none' and 'states' in obs:
      feat_r = sg(self.feat2tensor(outs['repfeat']))[:, :-1]
      act_r = jnp.concatenate(
          [prevact[k][:, 1:] for k in sorted(prevact)], -1)
      ens_preds_r = self.ensemble(feat_r, act_r)
      disagree_r = expl.disagreement(ens_preds_r)  # (B, T-1)
      metrics['exploration/report_disagree_mean'] = disagree_r.mean()
      metrics['exploration/report_disagree_std'] = disagree_r.std()
      metrics['_heatmap_disagree'] = disagree_r
      metrics['_heatmap_states'] = obs['states']

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  @staticmethod
  def _compute_token_dim(enc_space, enc_config):
    dim = 0
    veckeys = [k for k, s in enc_space.items() if len(s.shape) <= 2]
    imgkeys = [k for k, s in enc_space.items() if len(s.shape) == 3]
    if veckeys:
      dim += enc_config.units
    if imgkeys:
      mults = enc_config.mults
      depth = enc_config.depth
      img_shape = list(enc_space[sorted(imgkeys)[0]].shape)
      h, w = img_shape[0], img_shape[1]
      for _ in mults:
        h, w = h // 2, w // 2
      dim += depth * mults[-1] * h * w
    return dim

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
    mode2=False,
    v_imag=None,
    priorval_imag=None,
    budget_d=0.1,
    gamma_cost=0.999,
    V_norm=88.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con

  if mode2:
    # SOOPER Mode 2: planning-MDP termination. Evaluate the in-imagination
    # switching condition Φ = c_<t + γ^t·q_pess at every imagined latent and
    # truncate the explorer's credit there, bootstrapping with V^pi~_r.
    #   cost_t   = 1 - con  (per-step termination prob — imagination analog
    #              of the binary hole cost log/hole_terminated in Mode 1)
    #   q_pess_t = clip(1 - v_imag/V_norm)  — distilled risk_critic, recovered
    #              from the V_prior head via the same clip the real gate
    #              applies on real V_prior. (Distilling V_prior directly
    #              avoids the zero-mass imbalance of risk_critic targets.)
    #              λ·σ omitted (λ=0).
    #   t        = imagination step index. Over H≈15 with gamma_cost≈0.999
    #              the γ^t decay is <2%, so using imagination-relative t
    #              (the real step_in_ep isn't available here) is fine.
    risk_imag = jnp.clip(1.0 - v_imag / V_norm, 0.0, 1.0)
    t_idx = jnp.arange(con.shape[1], dtype=f32)
    gamma_t = (gamma_cost ** t_idx)[None, :]
    disc_cost = gamma_t * (1.0 - con)
    c_lt = jnp.cumsum(disc_cost, 1) - disc_cost            # exclusive cumsum
    phi = c_lt + gamma_t * risk_imag
    switched = (jnp.cumsum((phi >= budget_d).astype(f32), 1) > 0).astype(f32)
    # r̃(s_t): at/after the switch the explorer's reward becomes V^pi~_r.
    # Only the first switched step's value actually feeds the return — later
    # steps are masked out of the loss by `weight` below.
    rew = jnp.where(switched > 0, priorval_imag, rew)
    # Treat the switch as a terminal event so lambda_return stops
    # accumulating future reward there and picks up V^pi~_r as the bootstrap.
    term = jnp.maximum(term, switched)
    # Truncate the actor's (and value's) credit at/after the switch — the
    # explorer's actions past the switch are not executed (prior is driving).
    weight = weight * (1.0 - switched)
    metrics['sooper/switch_frac'] = switched.mean()
    metrics['sooper/phi_mean'] = phi.mean()
    metrics['sooper/v_imag_mean'] = v_imag.mean()
    metrics['sooper/risk_imag_mean'] = risk_imag.mean()
    metrics['sooper/priorval_imag_mean'] = priorval_imag.mean()

  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
