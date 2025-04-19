#!/usr/bin/env python3
"""
Simple script to visualize a trained policy.
"""

import os
import argparse
import jax
import jax.numpy as jp
import mediapy as media
from brax.io import html, model
from IPython.display import HTML, display
import functools

from environments import get_environment
from brax.training.agents.ppo import train as ppo


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--env", type=str, default="quadruped", help="Environment name")
    parser.add_argument("--scene", type=str, default=None, help="Scene file (for quadruped)")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to render")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output video file (optional)")
    parser.add_argument("--use_brax_renderer", action="store_true", help="Use Brax HTML renderer")
    
    # Quadruped specific args
    parser.add_argument("--x_vel", type=float, default=1.0, help="Target x velocity")
    parser.add_argument("--y_vel", type=float, default=0.0, help="Target y velocity")
    parser.add_argument("--ang_vel", type=float, default=0.0, help="Target angular velocity")
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    
    print(f"Loading policy from: {checkpoint_path}")
    print(f"JAX devices: {jax.devices()}")
    
    # Create environment
    env_kwargs = {}
    if args.env == "quadruped" and args.scene is not None:
        env_kwargs["scene_file"] = args.scene
    
    eval_env = get_environment(args.env, **env_kwargs)
    
    # Load policy using the approach from the example
    print(f"Loading policy from: {checkpoint_path}")
    
    # Get the make_inference_fn by creating a dummy training function with 0 timesteps
    make_inference_fn, _, _ = ppo.train(environment=eval_env, num_timesteps=0)
    
    # Load the parameters
    params = model.load_params(checkpoint_path)
    
    # Create the inference function
    inference_fn = make_inference_fn(params)
    print("Created inference function")
    
    # JIT the environment functions
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(inference_fn)
    
    # Set command for quadruped
    the_command = jp.array([args.x_vel, args.y_vel, args.ang_vel])
    print(f"Setting command: x_vel={args.x_vel}, y_vel={args.y_vel}, ang_vel={args.ang_vel}")
    
    # Initialize the environment
    rng = jax.random.PRNGKey(args.seed)
    state = jit_reset(rng)
    
    # Set command for quadruped
    if args.env == "quadruped" and hasattr(state, 'info') and 'command' in state.info:
        state.info['command'] = the_command
    
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
    
    if args.use_brax_renderer:
        # Use Brax HTML renderer
        html_str = html.render(eval_env.sys.tree_replace({'opt.timestep': eval_env.dt}), rollout)
        
        # Save HTML to file if output is specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write(html_str)
            print(f"HTML saved to {args.output}")
        else:
            # Try to display in notebook if possible
            try:
                display(HTML(html_str))
            except:
                print("Cannot display HTML in this environment.")
                print("Run with --output to save HTML to a file.")
    else:
        # Use mediapy renderer
        try:
            frames = eval_env.render(rollout[::render_every], camera='track')
            
            # Save or display video
            if args.output:
                media.write_video(args.output, frames, fps=1.0 / eval_env.dt / render_every)
                print(f"Video saved to {args.output}")
            else:
                media.show_video(frames, fps=1.0 / eval_env.dt / render_every)
        except Exception as e:
            print(f"Error rendering: {e}")
            print("This might be due to missing ffmpeg. Try installing it with:")
            print("  apt-get install ffmpeg  # On Ubuntu/Debian")
            print("  brew install ffmpeg     # On macOS")


if __name__ == "__main__":
    main() 