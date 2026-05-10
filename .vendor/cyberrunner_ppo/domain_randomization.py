"""Domain randomization for physics parameters.

Applies ±10% randomization to nominal values from the model on each reset.
All nominal values are defined in builder.py and this module simply applies
relative randomization without maintaining separate range configurations.
"""

import jax
import jax.numpy as jnp
from mujoco import mjx
from functools import partial
from typing import Dict, Any, Tuple, Optional


# Default randomization percentage (±10%)
DEFAULT_RANDOMIZATION_PERCENT = 0.15


class DomainRandomizer:
    """Randomize physics parameters using JAX.

    Applies ±10% randomization (by default) to the following parameters:
    - Actuator gear (alpha, beta)
    - Actuator dynamics time constant (alpha, beta)
    - Joint damping (alpha, beta)
    - Joint frictionloss (alpha, beta)
    - Marble friction (slide, spin, roll)
    - Marble mass
    - Marble solref dampingratio

    All nominal values come from the compiled model. This class applies
    relative randomization without maintaining separate configuration.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        randomization_percent: float = DEFAULT_RANDOMIZATION_PERCENT
    ):
        """Initialize domain randomizer.

        Args:
            config: Optional config dict (for backwards compatibility, mostly ignored)
            randomization_percent: Percentage to randomize (0.10 = ±10%)
        """
        self.randomization_percent = randomization_percent

        # Allow config to override randomization percent if provided
        if isinstance(config, dict):
            self.randomization_percent = config.get(
                'randomization_percent',
                randomization_percent
            )

    @partial(jax.jit, static_argnums=(0,))
    def randomize(
        self,
        model: mjx.Model,
        rng: jax.random.PRNGKey
    ) -> mjx.Model:
        """Apply randomization to model.

        Args:
            model: MJX model to randomize
            rng: JAX random key

        Returns:
            Randomized model (new instance, original unchanged)
        """
        keys = jax.random.split(rng, 10)

        pct = self.randomization_percent

        # Randomize actuator gear (indices 0 and 1 for alpha and beta)
        model = self._randomize_actuator_gear(model, keys[0], pct)

        # Randomize actuator dynamics (dynprm tau)
        model = self._randomize_actuator_dynprm(model, keys[1], pct)

        # Randomize joint damping
        model = self._randomize_joint_damping(model, keys[2], pct)

        # Randomize joint frictionloss
        model = self._randomize_joint_frictionloss(model, keys[3], pct)

        # Randomize marble friction (slide, spin, roll)
        model = self._randomize_marble_friction(model, keys[4], pct)

        # Randomize marble mass
        model = self._randomize_marble_mass(model, keys[5], pct)

        # Randomize marble solref dampingratio
        model = self._randomize_marble_solref(model, keys[6], pct)

        return model

    def _sample_scale(
        self,
        rng: jax.random.PRNGKey,
        pct: float,
        shape: tuple = ()
    ) -> jnp.ndarray:
        """Sample a random scale factor in range [1-pct, 1+pct].

        Args:
            rng: JAX random key
            pct: Randomization percentage (e.g., 0.10 for ±10%)
            shape: Shape of output array

        Returns:
            Scale factor(s) to multiply with nominal value
        """
        return jax.random.uniform(
            rng,
            shape=shape,
            minval=1.0 - pct,
            maxval=1.0 + pct
        )

    def _randomize_actuator_gear(
        self,
        model: mjx.Model,
        rng: jax.random.PRNGKey,
        pct: float
    ) -> mjx.Model:
        """Randomize actuator gear ratios (alpha and beta).

        Actuator gear is stored in model.actuator_gear with shape [nu, 6].
        We only modify the first element of each actuator's gear vector.
        """
        if model.nu < 2:
            return model

        keys = jax.random.split(rng, 2)

        # Get current gear values
        gear = model.actuator_gear

        # Randomize alpha (actuator 0) gear[0, 0]
        scale_alpha = self._sample_scale(keys[0], pct)
        gear = gear.at[0, 0].multiply(scale_alpha)

        # Randomize beta (actuator 1) gear[1, 0]
        scale_beta = self._sample_scale(keys[1], pct)
        gear = gear.at[1, 0].multiply(scale_beta)

        return model.replace(actuator_gear=gear)

    def _randomize_actuator_dynprm(
        self,
        model: mjx.Model,
        rng: jax.random.PRNGKey,
        pct: float
    ) -> mjx.Model:
        """Randomize actuator dynamics parameters (tau for filter dynamics).

        For filter dynamics, dynprm[0] is the time constant tau.
        """
        if model.nu < 2:
            return model

        keys = jax.random.split(rng, 2)

        # Get current dynprm values
        dynprm = model.actuator_dynprm

        # Randomize alpha (actuator 0) dynprm[0, 0] (tau)
        scale_alpha = self._sample_scale(keys[0], pct)
        dynprm = dynprm.at[0, 0].multiply(scale_alpha)

        # Randomize beta (actuator 1) dynprm[1, 0] (tau)
        scale_beta = self._sample_scale(keys[1], pct)
        dynprm = dynprm.at[1, 0].multiply(scale_beta)

        return model.replace(actuator_dynprm=dynprm)

    def _randomize_joint_damping(
        self,
        model: mjx.Model,
        rng: jax.random.PRNGKey,
        pct: float
    ) -> mjx.Model:
        """Randomize joint damping for alpha (DOF 0) and beta (DOF 1)."""
        keys = jax.random.split(rng, 2)

        # Get current damping values
        damping = model.dof_damping

        # Randomize alpha damping (DOF 0)
        scale_alpha = self._sample_scale(keys[0], pct)
        damping = damping.at[0].multiply(scale_alpha)

        # Randomize beta damping (DOF 1)
        scale_beta = self._sample_scale(keys[1], pct)
        damping = damping.at[1].multiply(scale_beta)

        return model.replace(dof_damping=damping)

    def _randomize_joint_frictionloss(
        self,
        model: mjx.Model,
        rng: jax.random.PRNGKey,
        pct: float
    ) -> mjx.Model:
        """Randomize joint frictionloss for alpha (DOF 0) and beta (DOF 1)."""
        keys = jax.random.split(rng, 2)

        # Get current frictionloss values
        frictionloss = model.dof_frictionloss

        # Randomize alpha frictionloss (DOF 0)
        scale_alpha = self._sample_scale(keys[0], pct)
        frictionloss = frictionloss.at[0].multiply(scale_alpha)

        # Randomize beta frictionloss (DOF 1)
        scale_beta = self._sample_scale(keys[1], pct)
        frictionloss = frictionloss.at[1].multiply(scale_beta)

        return model.replace(dof_frictionloss=frictionloss)

    def _randomize_marble_friction(
        self,
        model: mjx.Model,
        rng: jax.random.PRNGKey,
        pct: float
    ) -> mjx.Model:
        """Randomize marble friction coefficients (slide, spin, roll).

        Marble geoms are the last geom(s) in the model.
        geom_friction has shape [ngeom, 3] with [slide, spin, roll].
        """
        keys = jax.random.split(rng, 3)

        # Marble geoms are at the end (after board, walls, etc.)
        # Assuming marble is the last geom for single marble case
        num_marbles = model.nbody - 4  # Subtract world, base, link, board
        if num_marbles <= 0:
            num_marbles = 1
        marble_geom_start = model.ngeom - num_marbles

        # Get current friction values
        friction = model.geom_friction

        # Randomize slide friction
        scale_slide = self._sample_scale(keys[0], pct)
        friction = friction.at[marble_geom_start:, 0].multiply(scale_slide)

        # Randomize spin friction
        scale_spin = self._sample_scale(keys[1], pct)
        friction = friction.at[marble_geom_start:, 1].multiply(scale_spin)

        # Randomize roll friction
        scale_roll = self._sample_scale(keys[2], pct)
        friction = friction.at[marble_geom_start:, 2].multiply(scale_roll)

        return model.replace(geom_friction=friction)

    def _randomize_marble_mass(
        self,
        model: mjx.Model,
        rng: jax.random.PRNGKey,
        pct: float
    ) -> mjx.Model:
        """Randomize marble mass and update inertia accordingly.

        Marble bodies are at the end of the body list.
        """
        # Marble bodies start after world(0), base(1), link(2), board(3)
        marble_body_start = 4
        num_marbles = model.nbody - marble_body_start
        if num_marbles <= 0:
            return model

        scale = self._sample_scale(rng, pct)

        # Update body mass
        body_mass = model.body_mass.at[marble_body_start:].multiply(scale)

        # Update body inertia (scales linearly with mass for same geometry)
        body_inertia = model.body_inertia.at[marble_body_start:, :].multiply(scale)

        return model.replace(
            body_mass=body_mass,
            body_inertia=body_inertia
        )

    def _randomize_marble_solref(
        self,
        model: mjx.Model,
        rng: jax.random.PRNGKey,
        pct: float
    ) -> mjx.Model:
        """Randomize marble solref dampingratio.

        solref has shape [ngeom, 2] with [timeconst, dampingratio].
        We randomize dampingratio (index 1).
        """
        # Marble geoms are at the end
        num_marbles = model.nbody - 4
        if num_marbles <= 0:
            num_marbles = 1
        marble_geom_start = model.ngeom - num_marbles

        scale = self._sample_scale(rng, pct)

        # Get current solref values
        solref = model.geom_solref

        # Randomize dampingratio (index 1)
        solref = solref.at[marble_geom_start:, 1].multiply(scale)

        return model.replace(geom_solref=solref)


def sample_physics_parameters(
    rng: jax.random.PRNGKey,
    model: mjx.Model,
    randomization_percent: float = DEFAULT_RANDOMIZATION_PERCENT
) -> Dict[str, float]:
    """Sample physics parameters for logging/analysis.

    This is useful for tracking what parameters were used in each episode.

    Args:
        rng: JAX random key
        model: MJX model (to get nominal values)
        randomization_percent: Percentage to randomize

    Returns:
        Dictionary of sampled parameters
    """
    keys = jax.random.split(rng, 10)
    pct = randomization_percent

    def sample_scale(key):
        return jax.random.uniform(key, minval=1.0-pct, maxval=1.0+pct)

    params = {}

    # Actuator parameters
    if model.nu >= 2:
        params['gear_alpha'] = model.actuator_gear[0, 0] * sample_scale(keys[0])
        params['gear_beta'] = model.actuator_gear[1, 0] * sample_scale(keys[1])
        params['dynprm_tau_alpha'] = model.actuator_dynprm[0, 0] * sample_scale(keys[2])
        params['dynprm_tau_beta'] = model.actuator_dynprm[1, 0] * sample_scale(keys[3])

    # Joint parameters
    params['damping_alpha'] = model.dof_damping[0] * sample_scale(keys[4])
    params['damping_beta'] = model.dof_damping[1] * sample_scale(keys[5])
    params['frictionloss_alpha'] = model.dof_frictionloss[0] * sample_scale(keys[6])
    params['frictionloss_beta'] = model.dof_frictionloss[1] * sample_scale(keys[7])

    # Marble parameters (assuming last geom)
    marble_geom_idx = model.ngeom - 1
    params['marble_friction_slide'] = model.geom_friction[marble_geom_idx, 0] * sample_scale(keys[8])
    params['marble_friction_spin'] = model.geom_friction[marble_geom_idx, 1] * sample_scale(keys[8])
    params['marble_friction_roll'] = model.geom_friction[marble_geom_idx, 2] * sample_scale(keys[8])

    # Marble mass (assuming last body is marble)
    marble_body_idx = model.nbody - 1
    params['marble_mass'] = model.body_mass[marble_body_idx] * sample_scale(keys[9])

    # Marble solref dampingratio
    params['marble_solref_dampingratio'] = model.geom_solref[marble_geom_idx, 1] * sample_scale(keys[9])

    return params