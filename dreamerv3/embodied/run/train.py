import collections
import time
from functools import partial as bind

import elements
import embodied
import numpy as np

try:
  from embodied.run import run_metrics
except ImportError:  # laptop driver / alternate sys.path layout
  from . import run_metrics


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
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  # Faithful, budget-normalized metrics (replaces the window-averaged epstats
  # + the old spatial _CoverageTracker). See embodied/run/run_metrics.py.
  layout_name = getattr(args, 'cyberrunner_layout', 'hard')
  exploration_stats = run_metrics.ExplorationStats(layout=layout_name)
  safety_stats = run_metrics.SafetyStats()
  gate_stats = run_metrics.GateStats()
  trigger_clips = run_metrics.TriggerClips(num_envs=args.envs)
  video_ep = []
  completed_video = [None]
  MAX_VIDEO_STEPS = 3000

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    is_first = bool(tran['is_first'])
    is_last = bool(tran['is_last'])
    is_first and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')

    # Find this env's rendered frame (uint8 HxWxC) once; reused for video.
    frame = None
    for key, value in tran.items():
      if getattr(value, 'dtype', None) == np.uint8 and getattr(value, 'ndim', 0) == 3:
        frame = value
        break

    # Faithful metrics.
    states = tran.get('states', None)
    pp = tran.get('log/path_progress', np.float32(0.0))
    exploration_stats.update(
        worker, states, pp, is_first, is_last, env_step=int(step))
    safety_stats.update(tran, is_last)
    gate_present = run_metrics.GateStats.present(tran)
    if gate_present:
      gate_stats.update(tran, worker, is_first, is_last)
      triggered = float(tran.get('log/gate/triggered', 0.0)) > 0.5
      trigger_clips.update(worker, frame, triggered, is_last)

    # Rollout video: gate-trigger clips when the gate is on; otherwise a
    # worker-0 full-episode video so plain OPAX still gets a rollout.
    if not gate_present and worker == 0:
      if is_first:
        video_ep.clear()
      if frame is not None and len(video_ep) < MAX_VIDEO_STEPS:
        video_ep.append(frame)
      if is_last and len(video_ep) > 10:
        completed_video[0] = np.stack(video_ep[:MAX_VIDEO_STEPS])
        video_ep.clear()

    if is_last:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')

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
    # Per-step calibration dump. When sooper.dump_steps=true, write one
    # jsonl line per env per control step into the run's logdir. Plain
    # str(...) so PolicySwitcher (no elements.Path dep) can open() it.
    dump_path = None
    if bool(getattr(sooper_cfg, 'dump_steps', False)):
      dump_path = str(logdir / 'sooper_steps.jsonl')
      print(f'[sooper] dump_steps enabled → {dump_path}', flush=True)
    policy = PolicySwitcher(
        agent, prior_fn, adapter, sooper_cfg,
        risk_source=risk_source, value_fn=value_fn,
        dump_path=dump_path,
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
          # states[2:4] is BOARD-FRAME normalized: [-1, 1] (centered at
          # board center, divided by board/2). viz.{sigma,coverage}_heatmap
          # expect corner-origin physical coords in [0, BOARD_W/H]. Shift
          # then scale at the boundary so viz.py stays in physical units.
          half = np.array(
              [viz.BOARD_WIDTH / 2.0, viz.BOARD_HEIGHT / 2.0],
              dtype=np.float32)
          ball_xy = (states[:, 1:, 2:4].reshape(-1, 2) + 1.0) * half
          ball_xy_all = (states[:, :, 2:4].reshape(-1, 2) + 1.0) * half
          result['exploration/sigma_heatmap'] = viz.sigma_heatmap(
              disagree, ball_xy)
          result['exploration/coverage_heatmap'] = viz.coverage_heatmap(
              ball_xy_all)
        except Exception as e:
          print(f'Heatmap generation failed: {e}')
      result.pop('_heatmap_disagree', None)
      result.pop('_heatmap_states', None)
      log_video = getattr(args, 'log_video', True)
      for key in list(result):
        v = np.asarray(result[key]) if hasattr(result[key], 'shape') else result[key]
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3:
          # Heatmap wrapped as single-frame video — drop entirely if
          # videos are off, otherwise keep wandb.Video happy with (1,H,W,3).
          if not log_video:
            result.pop(key)
          else:
            result[key] = v[None]
        elif isinstance(v, np.ndarray) and v.ndim == 4 and v.shape[-1] == 1:
          if not log_video:
            result.pop(key)
          else:
            result[key] = np.repeat(v, 3, axis=-1)
        elif isinstance(v, np.ndarray) and v.ndim >= 4 and not log_video:
          # Any other multi-frame video (e.g. openloop comparisons).
          result.pop(key)
      logger.add(result, prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})

      # Faithful safety + exploration metrics (cumulative counters, not
      # window-averaged). Gate metrics only emit on SOOPER runs.
      logger.add(safety_stats.emit())
      logger.add(exploration_stats.emit())
      if gate_stats.triggers > 0 or gate_stats.prior_steps > 0:
        logger.add(gate_stats.emit())
        cov = exploration_stats.emit()['exploration/coverage']
        holes = safety_stats.holes
        logger.add({'pareto/coverage_per_fall':
                    cov / holes if holes > 0 else float(cov)})

      # GPU memory
      logger.add(_gpu_memory_stats())

      # Rollout video. Off-switch via --run.log_video false. For SOOPER runs
      # we log reservoir-sampled gate-trigger clips (~1 s before / ~2 s after
      # each trigger); for plain OPAX we log a worker-0 full-episode video.
      if getattr(args, 'log_video', True):
        for name, clip in trigger_clips.drain().items():
          if clip.shape[-1] == 1:
            clip = np.repeat(clip, 3, axis=-1)
          logger.add({name: clip})
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
