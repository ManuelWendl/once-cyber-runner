"""Thin wrapper around MaxInfoSAC to fix SB3 save/load compatibility.

SB3's BaseAlgorithm.load() calls cls(policy, env, device, _init_setup_model=False)
without passing ensemble_model_kwargs, which is required by MaxInfoSAC.__init__.
Additionally, MaxInfoSAC._setup_ensemble_model runs in __init__ (not gated by
_init_setup_model), so we must skip it when ensemble_model_kwargs is empty (load path).
The ensemble state is fully restored from the saved __dict__ by SB3's load().
"""
import copy
from stable_baselines3.sac import SAC
from maxinforl_torch.commons.maxinfo_sac import MaxInfoSAC as _MaxInfoSAC
from typing import Dict, Optional


class MaxInfoSAC(_MaxInfoSAC):
    def __init__(self, ensemble_model_kwargs: Optional[Dict] = None, *args, **kwargs):
        if ensemble_model_kwargs is None:
            ensemble_model_kwargs = {}
        self._skip_ensemble_setup = len(ensemble_model_kwargs) == 0
        super().__init__(ensemble_model_kwargs=ensemble_model_kwargs, *args, **kwargs)

    def _setup_ensemble_model(self, **kwargs):
        if self._skip_ensemble_setup:
            return
        super()._setup_ensemble_model(**kwargs)

    def _setup_model(self) -> None:
        # On load path: dyn_entropy_scale is None (was 'auto', set to None during
        # original training). The saved log_dyn_entropy_scale and optimizer will be
        # restored from __dict__, so skip the dyn_entropy_scale branch entirely.
        if self.dyn_entropy_scale is None:
            # Call grandparent (SAC) _setup_model, then just create actor_target
            SAC._setup_model(self)
            self.actor_target = copy.deepcopy(self.actor)
            # dyn_entropy_scale stays None, log_dyn_entropy_scale and optimizer
            # will be restored from saved state
        else:
            super()._setup_model()
