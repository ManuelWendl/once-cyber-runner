from . import parallel, wrappers


def make_envs(config):
    def env_constructor(idx):
        return lambda: make_env(config, idx)

    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    eval_envs = parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.gray,
            noops=config.noops,
            lives=config.lives,
            sticky=config.sticky,
            actions=config.actions,
            length=config.time_limit,
            pooling=config.pooling,
            aggregate=config.aggregate,
            resize=config.resize,
            autostart=config.autostart,
            clip_reward=config.clip_reward,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        import envs.metaworld as metaworld

        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
        )
    elif suite == "cyberrunner":
        import envs.cyberrunner as cyberrunner

        env = cyberrunner.CyberRunner(
            task,
            config.action_repeat,
            config.size,
            config.seed + id,
            reward_every_n_waypoints=config.reward_every_n_waypoints,
            hole_penalty=config.hole_penalty,
            checkpoint_radius=config.checkpoint_radius,
            checkpoint_hold_steps=config.checkpoint_hold_steps,
            checkpoint_speed_threshold=config.checkpoint_speed_threshold,
            checkpoint_arrival_reward=config.checkpoint_arrival_reward,
            checkpoint_stabilize_reward=config.checkpoint_stabilize_reward,
            checkpoint_hold_reward=config.checkpoint_hold_reward,
            safe_hole_margin=config.safe_hole_margin,
            checkpoint_speed_ema_alpha=config.checkpoint_speed_ema_alpha,
            prior_mode=getattr(config, "prior_mode", False),
            prior_start_waypoint_window=getattr(config, "prior_start_waypoint_window", 3),
            checkpoint_progress_reward_scale=getattr(config, "checkpoint_progress_reward_scale", 20.0),
            terminate_on_checkpoint_stabilized=getattr(config, "terminate_on_checkpoint_stabilized", False),
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)
