#!/usr/bin/env python3
"""
Simple script to load and visualize a trained policy.
"""

import os
import argparse
import jax
import jax.numpy as jp
import mediapy as media

from environments import get_environment
from policies import make_inference_fn
from utils.checkpoint import load_params


def main():
    parser = argparse.ArgumentParser(description="Load and visualize a trained policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--env", type=str, default="quadruped", help="Environment name")
    parser.add_argument("--scene", type=str, default=None, help="Scene file (for quadruped)")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to render")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output video file (optional)")
    
    # Quadruped specific args
    parser.add_argument("--x_vel", type=float, default=1.0, help="Target x velocity")
    parser.add_argument("--y_vel", type=float, default=0.0, help="Target y velocity")
    parser.add_argument("--ang_vel", type=float, default=0.0, help="Target angular velocity")
    
    args = parser.parse_args()
    
    print(f"Loading policy from: {args.checkpoint}")
    print(f"JAX devices: {jax.devices()}")
    
    # Create environment
    env_kwargs = {}
    if args.env == "quadruped" and args.scene is not None:
        env_kwargs["scene_file"] = args.scene
    
    env = get_environment(args.env, **env_kwargs)
    
    # Load policy
    params = load_params(args.checkpoint)
    inference_fn = make_inference_fn()(params)
    
    # JIT the environment functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)
    
    # Initialize the environment
    rng = jax.random.PRNGKey(args.seed)
    state = jit_reset(rng)
    
    # Set command for quadruped
    if args.env == "quadruped" and hasattr(state, 'info') and 'command' in state.info:
        command = jp.array([args.x_vel, args.y_vel, args.ang_vel])
        state.info['command'] = command
        print(f"Setting command: x_vel={args.x_vel}, y_vel={args.y_vel}, ang_vel={args.ang_vel}")
    
    # Collect trajectory
    rollout = [state.pipeline_state]
    rewards = []
    
    for i in range(args.steps):
        act_rng, rng = jax.random.split(rng)
        action, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)
        rewards.append(float(state.reward))
    
    # Print statistics
    print(f"Average reward: {sum(rewards) / len(rewards):.2f}")
    print(f"Total reward: {sum(rewards):.2f}")
    
    # Render the trajectory
    render_every = 2
    try:
        frames = env.render(rollout[::render_every], camera='track')
        
        # Save or display video
        if args.output:
            media.write_video(args.output, frames, fps=1.0 / env.dt / render_every)
            print(f"Video saved to {args.output}")
        else:
            media.show_video(frames, fps=1.0 / env.dt / render_every)
    except Exception as e:
        print(f"Error rendering: {e}")
        print("This might be due to missing ffmpeg. Try installing it with:")
        print("  apt-get install ffmpeg  # On Ubuntu/Debian")
        print("  brew install ffmpeg     # On macOS")


if __name__ == "__main__":
    main() 