# Prior Integration Plan

## Goal

Train a local stabilization prior and integrate it into the current R2-Dreamer Cyberrunner pipeline in a SOOPER-style way:

- the explorer expands the frontier
- the prior recovers and stabilizes locally
- later, the world model becomes aware of safety/fallback structure

This plan assumes the prior is trained first in simulation on the local checkpoint stabilization task.

## Phase 1: Train And Validate The Prior

### Objective

Train a policy that:

- starts near a safe checkpoint
- moves toward that checkpoint
- reduces speed
- stabilizes the ball there
- avoids holes

### Current Setup

This is implemented through:

- `prior_mode=True` in `envs/cyberrunner.py`
- `configs/env/cyberrunner_prior_state.yaml`
- checkpoint-conditioned observation `checkpoint`
- dense reward toward the checkpoint
- success termination on stabilization

### Metrics To Track

The prior is considered promising when these improve:

- `episode/success`
- `episode/stable_steps`
- `episode/checkpoint_dist`
- `episode/ball_speed`
- `episode/score`

### Acceptance Criteria

Do not integrate the prior until all of these are reasonably satisfied:

- high success rate on local stabilization episodes
- low hole rate
- stable performance across multiple checkpoint targets
- consistent settling under noisy observations

### Immediate Follow-Up

Before integration, run dedicated eval episodes for:

- different checkpoint targets
- small perturbations in initial position
- different seeds

## Phase 2: Freeze The Prior

### Objective

Once the prior is good enough, save it and keep it fixed.

### Why

The prior should be the trusted conservative policy. During explorer training it should not keep changing, otherwise the fallback boundary becomes unstable.

### Deliverables

- saved prior checkpoint
- clear config used to train it
- small evaluation summary

## Phase 3: Add A Rule-Based Fallback Gate

### Objective

Use the prior during exploration only when needed.

### First Integration Strategy

Implement a rule-based action override:

- Dreamer proposes the default action
- if a safety trigger fires, replace that action with the prior action

### Good Initial Safety Triggers

- `safe_hole_margin < threshold`
- speed too high near a checkpoint
- inside checkpoint basin but failing to stabilize
- optional timeout while trying to stabilize

### Suggested First Placement

Add this switch at the actor/environment interface, not inside the world model yet.

Likely integration points:

- `trainer.py`
- or a small policy wrapper around `agent.act(...)`

### Metrics To Compare

Evaluate:

- plain Dreamer
- Dreamer + prior fallback

Track:

- hole rate
- unlocked checkpoints
- stable steps
- path progress
- episode score

### Success Criterion

If fallback improves safety and checkpoint progression without collapsing exploration, move to training with fallback enabled.

## Phase 4: Train Dreamer With Fallback Active

### Objective

Collect data using the hybrid controller:

- Dreamer for exploration
- prior for local rescue/stabilization

### Expected Effect

Replay now contains safer trajectories and more successful stabilization events.

This should improve:

- world model coverage of safe local dynamics
- actor learning around frontier checkpoints
- sample efficiency for safe exploration

### Comparison To Run

Train two versions:

1. Dreamer only
2. Dreamer + prior fallback

Compare:

- unlocked checkpoint index
- stable steps
- hole rate
- episode length
- final maze success

## Phase 5: Add World-Model Safety Awareness

### Objective

Make the world model explicitly aware of safety and fallback.

### First Extension

Add a latent risk/cost head in `dreamer.py` that predicts:

- near-hole risk
- failure probability
- or prior takeover probability

### Use In Imagination

Then modify imagined reward or rollout logic so that:

- risky imagined trajectories are penalized
- or trajectories are truncated when fallback would be invoked

This is the first real step toward SOOPER-style prior-aware imagination.

## Phase 6: SOOPER-Style Training Structure

### Target Behavior

The final intended structure is:

- optimistic exploration in the safe/improving region
- pessimistic reliance on the prior near unsafe states
- progressive expansion of the trusted frontier

### Practical Version In This Repo

The likely order is:

1. prior-trained local stabilizer
2. explicit fallback gate
3. fallback-active data collection
4. risk-aware imagination
5. optional prior-triggered rollout truncation

## Immediate Next Steps

1. Make prior training robust.
   Fix reset/task construction so prior episodes are valid and learnable.

2. Validate prior performance.
   Confirm success rate and stabilization quality across checkpoints.

3. Save and freeze the prior.

4. Implement a simple fallback gate in the main exploration loop.

5. Run Dreamer vs Dreamer+prior comparison.

## Open Questions

- What threshold should trigger prior takeover?
- Should fallback persist for a fixed horizon or until stabilization completes?
- Should the prior target the current active checkpoint or the nearest safe checkpoint?
- When fallback is active, should the exploration policy still receive credit for the transition?
- Should imagined rollouts terminate when fallback would be invoked, or just receive a cost penalty?

## Suggested Implementation Order

1. Prior training cleanup and validation
2. Prior checkpoint saving/loading utility
3. Rule-based fallback wrapper
4. Hybrid eval experiments
5. Hybrid training experiments
6. Risk head in the world model
7. Prior-aware imagination
