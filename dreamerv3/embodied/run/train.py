import collections
import time
from functools import partial as bind

import elements
import embodied
import numpy as np


class _CoverageTracker:
  """Tracks ball positions and computes coverage / path progress stats."""

  def __init__(self, grid_res=30, board_w=0.276, board_h=0.231):
    self.grid_res = grid_res
    self.board_w = board_w
    self.board_h = board_h
    self.visit_counts = np.zeros((grid_res, grid_res), dtype=np.int64)
    self.max_progress_per_episode = []
    self._current_max = {}

  def update(self, worker, states, path_progress, is_first, is_last):
    if states.ndim < 1 or states.shape[-1] < 4:
      return
    bx, by = float(states[2]), float(states[3])
    cx = min(int(bx / self.board_w * self.grid_res), self.grid_res - 1)
    cy = min(int(by / self.board_h * self.grid_res), self.grid_res - 1)
    cx, cy = max(cx, 0), max(cy, 0)
    self.visit_counts[cy, cx] += 1

    pp = float(path_progress)
    if is_first:
      self._current_max[worker] = 0.0
    self._current_max[worker] = max(self._current_max.get(worker, 0.0), pp)
    if is_last:
      self.max_progress_per_episode.append(self._current_max.pop(worker, 0.0))

  def coverage(self):
    return float((self.visit_counts > 0).sum()) / self.visit_counts.size

  def entropy(self):
    total = self.visit_counts.sum()
    if total == 0:
      return 0.0
    p = self.visit_counts.ravel() / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

  def mean_max_path_progress(self, last_n=100):
    if not self.max_progress_per_episode:
      return 0.0
    return float(np.mean(self.max_progress_per_episode[-last_n:]))


def _gpu_memory_stats():
  stats = {}
  try:
    import subprocess
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total',
         '--format=csv,nounits,noheader'],
        timeout=5).decode().strip().split('\n')[0]
    used, total = out.split(',')
    stats['gpu/memory_used_mb'] = float(used.strip())
    stats['gpu/memory_total_mb'] = float(total.strip())
    stats['gpu/memory_pct'] = float(used.strip()) / max(float(total.strip()), 1) * 100
  except Exception:
    pass
  return stats


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  coverage_tracker = _CoverageTracker()
  video_ep = []
  completed_video = [None]
  MAX_VIDEO_STEPS = 2000

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    # Coverage tracking
    states = tran.get('states', None)
    pp = tran.get('log/path_progress', np.float32(0.0))
    if states is not None:
      coverage_tracker.update(
          worker, states, pp, bool(tran['is_first']), bool(tran['is_last']))

    # Collect video frames from worker 0 (full episode)
    if worker == 0:
      if tran['is_first']:
        video_ep.clear()
      if len(video_ep) < MAX_VIDEO_STEPS:
        for key, value in tran.items():
          if value.dtype == np.uint8 and value.ndim == 3:
            video_ep.append(value)
            break
      if tran['is_last'] and len(video_ep) > 10:
        completed_video[0] = np.stack(video_ep[:MAX_VIDEO_STEPS])
        video_ep.clear()

    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  start_time = time.time()
  print('Start training loop')
  # SOOPER fallback gate (off by default → plain OPAX). When sooper.enabled
  # is True, wrap the policy with PolicySwitcher: it runs OPAX, computes a
  # K-step risk_horizon from the continuation head, and routes control to
  # the survival prior under hysteresis. See dreamerv3/dreamerv3/sooper.py.
  sooper_cfg = getattr(args, 'sooper', None)
  if sooper_cfg is not None and bool(getattr(sooper_cfg, 'enabled', False)):
    # main.py rewrites sys.path so the *inner* dreamerv3/dreamerv3/ becomes
    # the top-level `dreamerv3` package. The laptop unit-test driver, by
    # contrast, adds the repo root and imports `dreamerv3.dreamerv3.sooper`.
    # Try both — first form for cluster runs, second for laptop.
    try:
      from dreamerv3.sooper import (
          PolicySwitcher, PriorObsAdapter,
          load_survival_prior, load_survival_prior_value, make_risk_source,
      )
    except ImportError:
      from dreamerv3.dreamerv3.sooper import (  # type: ignore[no-redef]
          PolicySwitcher, PriorObsAdapter,
          load_survival_prior, load_survival_prior_value, make_risk_source,
      )
    risk_mode = getattr(sooper_cfg, 'risk_mode', 'cont_product')
    print(f'[sooper] enabled — prior_pkl={sooper_cfg.prior_pkl} '
          f'risk_mode={risk_mode}', flush=True)
    prior_fn = load_survival_prior(sooper_cfg.prior_pkl)
    # Always load the critic — costs ~1 MLP forward per step but gives us
    # V_prior and risk_critic in the logs alongside the cont signals so a
    # single run produces comparable histograms for all three risk sources.
    value_fn = load_survival_prior_value(sooper_cfg.prior_pkl)
    risk_source = make_risk_source(risk_mode)
    adapter = PriorObsAdapter(num_envs=args.envs)
    policy = PolicySwitcher(
        agent, prior_fn, adapter, sooper_cfg,
        risk_source=risk_source, value_fn=value_fn,
    )
  else:
    policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      result = agg.result()
      if '_heatmap_disagree' in result and '_heatmap_states' in result:
        try:
          from dreamerv3 import viz
          disagree = np.asarray(result.pop('_heatmap_disagree')).reshape(-1)
          states = np.asarray(result.pop('_heatmap_states'))
          ball_xy = states[:, 1:, 2:4].reshape(-1, 2)
          ball_xy_all = states[:, :, 2:4].reshape(-1, 2)
          result['exploration/sigma_heatmap'] = viz.sigma_heatmap(
              disagree, ball_xy)
          result['exploration/coverage_heatmap'] = viz.coverage_heatmap(
              ball_xy_all)
        except Exception as e:
          print(f'Heatmap generation failed: {e}')
      result.pop('_heatmap_disagree', None)
      result.pop('_heatmap_states', None)
      for key in list(result):
        v = np.asarray(result[key]) if hasattr(result[key], 'shape') else result[key]
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3:
          result[key] = v[None]  # (H,W,3) -> (1,H,W,3) so logger treats as single-frame video
        elif isinstance(v, np.ndarray) and v.ndim == 4 and v.shape[-1] == 1:
          result[key] = np.repeat(v, 3, axis=-1)
      logger.add(result, prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})

      # Coverage and path progress
      logger.add({
          'exploration/coverage': coverage_tracker.coverage(),
          'exploration/visitation_entropy': coverage_tracker.entropy(),
          'exploration/mean_max_path_progress': coverage_tracker.mean_max_path_progress(),
      })

      # GPU memory
      logger.add(_gpu_memory_stats())

      # Policy rollout video (full episode from worker 0)
      if completed_video[0] is not None:
        vid = completed_video[0]
        if vid.shape[-1] == 1:
          vid = np.repeat(vid, 3, axis=-1)
        logger.add({'policy_video': vid})
        completed_video[0] = None

      # Ensure all video/image metrics have 3 channels (WandB can't encode 1-channel)
      for i, (s, name, value) in enumerate(logger._metrics):
        if hasattr(value, 'shape') and hasattr(value, 'ndim'):
          v = np.asarray(value)
          if v.ndim >= 3 and v.shape[-1] == 1:
            logger._metrics[i] = (s, name, np.repeat(v, 3, axis=-1))
      logger.write()
      elapsed = time.time() - start_time
      cur = int(step)
      total = int(args.steps)
      frac = cur / total if total > 0 else 0.0
      eta = elapsed * (1.0 - frac) / frac if frac > 0 else float('inf')
      def _fmt(s):
        if not np.isfinite(s):
          return '?'
        h, rem = divmod(int(s), 3600)
        m, s = divmod(rem, 60)
        return f'{h:d}h{m:02d}m{s:02d}s'
      print(
          f'[progress] step {cur}/{total} ({100*frac:5.2f}%)  '
          f'elapsed {_fmt(elapsed)}  ETA {_fmt(eta)}',
          flush=True,
      )

    if should_save(step):
      cp.save()

  logger.close()
