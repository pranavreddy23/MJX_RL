"""
Utilities for rendering MJX environments.
"""

import time
from typing import Callable, Optional, List

import jax
import jax.numpy as jp
import numpy as np
import mediapy as media

from brax.envs.base import Env
from brax.io import html


def render_policy(
    env: Env,
    inference_fn: Callable,
    n_steps: int = 500,
    render_every: int = 2,
    seed: int = 0,
    command: Optional[jp.ndarray] = None,
    fps: Optional[float] = None,
    camera: str = 'track',
    width: int = 480,
    height: int = 360,
    use_html: bool = False,
):
    """
    Render a policy in an environment.
    
    Args:
        env: Environment to render
        inference_fn: Function that takes (obs, rng) and returns (action, state)
        n_steps: Number of steps to render
        render_every: Render every N steps
        seed: Random seed
        command: Command to use (for environments that support it)
        fps: Frames per second (if None, uses 1.0 / env.dt / render_every)
        camera: Camera to use
        width: Width of the rendering
        height: Height of the rendering
        use_html: Whether to use HTML rendering (for notebooks)
    """
    # JIT the environment functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)
    
    # Initialize the environment
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)
    
    # Set command if provided
    if command is not None and hasattr(state, 'info') and 'command' in state.info:
        state.info['command'] = command
    
    # Collect trajectory
    rollout = [state.pipeline_state]
    
    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        action, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)
    
    # Render the trajectory
    if use_html:
        from IPython.display import display, HTML
        display(HTML(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout)))
    else:
        if fps is None:
            fps = 1.0 / env.dt / render_every
        
        frames = env.render(rollout[::render_every], camera=camera, width=width, height=height)
        media.show_video(frames, fps=fps)
    
    return rollout


def compare_policies(
    env: Env,
    inference_fns: List[Callable],
    labels: List[str],
    n_steps: int = 500,
    render_every: int = 2,
    seed: int = 0,
    command: Optional[jp.ndarray] = None,
    fps: Optional[float] = None,
    camera: str = 'track',
    width: int = 480,
    height: int = 360,
):
    """
    Compare multiple policies in the same environment.
    
    Args:
        env: Environment to render
        inference_fns: List of inference functions
        labels: Labels for each policy
        n_steps: Number of steps to render
        render_every: Render every N steps
        seed: Random seed
        command: Command to use (for environments that support it)
        fps: Frames per second (if None, uses 1.0 / env.dt / render_every)
        camera: Camera to use
        width: Width of the rendering
        height: Height of the rendering
    """
    assert len(inference_fns) == len(labels), "Number of inference functions must match number of labels"
    
    # JIT the environment functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fns = [jax.jit(fn) for fn in inference_fns]
    
    # Initialize the environment for each policy
    rng = jax.random.PRNGKey(seed)
    states = []
    rollouts = []
    
    for i in range(len(inference_fns)):
        policy_rng = jax.random.fold_in(rng, i)
        state = jit_reset(policy_rng)
        
        # Set command if provided
        if command is not None and hasattr(state, 'info') and 'command' in state.info:
            state.info['command'] = command
        
        states.append(state)
        rollouts.append([state.pipeline_state])
    
    # Collect trajectories
    for step in range(n_steps):
        for i, (state, jit_inference_fn) in enumerate(zip(states, jit_inference_fns)):
            policy_rng = jax.random.fold_in(rng, i * n_steps + step)
            action, _ = jit_inference_fn(state.obs, policy_rng)
            states[i] = jit_step(state, action)
            rollouts[i].append(states[i].pipeline_state)
    
    # Render each trajectory
    if fps is None:
        fps = 1.0 / env.dt / render_every
    
    for i, label in enumerate(labels):
        print(f"Rendering policy: {label}")
        frames = env.render(rollouts[i][::render_every], camera=camera, width=width, height=height)
        media.show_video(frames, fps=fps)
        time.sleep(1)  # Add a small delay between videos
    
    return rollouts 