"""
Humanoid environment implementation using MJX.
"""

import os
import jax
import jax.numpy as jp
from flax import struct
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional

import mujoco
from mujoco import mjx

from brax import base
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import mjcf
import etils.epath as epath

# Update the path to point to the humanoid folder at the project root
_HUMANOID_XML_PATH = epath.Path(__file__).parent.parent / 'humanoid' / 'humanoid.xml'

class Humanoid(PipelineEnv):
    """Environment for training the humanoid in MJX."""

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        """Initialize the humanoid environment.
        
        Args:
            forward_reward_weight: Weight for the forward velocity reward.
            ctrl_cost_weight: Weight for the control cost penalty.
            healthy_reward: Reward for staying healthy.
            terminate_when_unhealthy: Whether to terminate episodes when unhealthy.
            healthy_z_range: Range of z-positions considered healthy.
            reset_noise_scale: Scale of noise added during reset.
            exclude_current_positions_from_observation: Whether to exclude position data from observation.
        """
        if not os.path.exists(_HUMANOID_XML_PATH):
            raise FileNotFoundError(f"Humanoid XML file not found at: {_HUMANOID_XML_PATH}")
        
        print(f"Loading humanoid model from: {_HUMANOID_XML_PATH}")
        mj_model = mujoco.MjModel.from_xml_path(_HUMANOID_XML_PATH.as_posix())
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        
        # Store dt for use in step calculation
        self._dt = sys.opt.timestep * kwargs.get('n_frames', physics_steps_per_control_step)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self._dt
        forward_reward = self._forward_reward_weight * velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, action)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        
        # Create a new metrics dict starting with any existing keys
        metrics = {}
        if hasattr(state, 'metrics') and state.metrics is not None:
            # Start with all existing keys/values
            for key in state.metrics:
                # Copy over all existing metrics with zeros as default
                metrics[key] = state.metrics.get(key, 0.0)
        
        # Now update with new values
        metrics.update({
            'forward_reward': forward_reward,
            'reward_linvel': forward_reward,
            'reward_quadctrl': -ctrl_cost,
            'reward_alive': healthy_reward,
            'x_position': com_after[0],
            'y_position': com_after[1],
            'distance_from_origin': jp.linalg.norm(com_after),
            'x_velocity': velocity[0],
            'y_velocity': velocity[1],
        })

        return State(data, obs, reward, done, metrics, state.info)

    def _get_obs(
        self, data: mjx.Data, action: jp.ndarray
    ) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        position = data.qpos
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
        ])
        
    @property
    def dt(self):
        """Return the simulation timestep."""
        return self._dt
        
    @property
    def observation_size(self):
        """Returns the size of the observation."""
        dim = self.sys.nq + self.sys.nv
        if self._exclude_current_positions_from_observation:
            dim -= 2  # Exclude global x,y position
        dim += (self.sys.nbody - 1) * 10  # cinert (10 per body)
        dim += (self.sys.nbody - 1) * 6   # cvel (6 per body)
        dim += self.sys.nu                # actuator forces
        return (dim,)
    
    @property
    def action_size(self):
        """Returns the size of the action space."""
        return self.sys.nu 