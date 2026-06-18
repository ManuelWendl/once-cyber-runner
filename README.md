# Usage:

### PPO (default)
```bash
python train_simple.py
```

### SAC
```bash
python train_simple.py algo=sac
```

# Override any param on the fly
```bash
python train_simple.py algo=ppo algo.learning_rate=3e-4 total_timesteps=500000
python train_simple.py algo=sac env.hole_penalty=10.0
```


# Structure:
```
configs/
  config.yaml       ← top-level (total_timesteps, device, env knobs)
  algo/
    ppo.yaml        ← all PPO hyperparams
    sac.yaml        ← all SAC hyperparams
```