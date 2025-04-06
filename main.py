"""
Main entry point for MJX RL training.
This file provides a high-level interface for training and evaluating policies.
"""

import os
import argparse
import jax
import time
from datetime import datetime

from environments import get_environment
from policies import train_ppo, make_inference_fn
from utils.rendering import render_policy
from configs.default_configs import get_ppo_config


def parse_args():
    parser = argparse.ArgumentParser(description="MJX RL Training")
    parser.add_argument("--env", type=str, default="quadruped", help="Environment name")
    parser.add_argument("--scene", type=str, default=None, help="Scene file (for quadruped)")
    parser.add_argument("--timesteps", type=int, default=20_000_000, help="Number of training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel environments")
    parser.add_argument("--checkpoint_dir", type=str, default="/tmp/mjx_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation, no training")
    parser.add_argument("--render", action="store_true", help="Render policy after training")
    parser.add_argument("--domain_rand", action="store_true", help="Use domain randomization")
    parser.add_argument("--show_progress_plot", action="store_true", default=True, help="Show training progress plots")
    
    # Quadruped specific args
    parser.add_argument("--x_vel", type=float, default=1.0, help="Target x velocity for quadruped")
    parser.add_argument("--y_vel", type=float, default=0.0, help="Target y velocity for quadruped")
    parser.add_argument("--ang_vel", type=float, default=0.0, help="Target angular velocity for quadruped")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"JAX devices: {jax.devices()}")
    
    # Create environment
    env_kwargs = {}
    if args.env == "quadruped" and args.scene is not None:
        env_kwargs["scene_file"] = args.scene
    
    env = get_environment(args.env, **env_kwargs)
    eval_env = get_environment(args.env, **env_kwargs)
    
    # Training or loading policy
    if not args.eval_only:
        print(f"Training {args.env} with {args.num_envs} parallel environments")
        start_time = datetime.now()
        
        # Configure training
        config = get_ppo_config()
        config.num_timesteps = args.timesteps
        config.num_envs = args.num_envs
        config.show_progress_plot = args.show_progress_plot

        # Extract network parameters to top level for compatibility
        config.policy_hidden_layer_sizes = config.network.policy_hidden_layer_sizes

        train_kwargs = {
            "config": config,
            "seed": args.seed,
            "checkpoint_dir": args.checkpoint_dir,
        }
        
        if args.domain_rand:
            train_kwargs["use_domain_randomization"] = True
            
        if args.load_checkpoint:
            train_kwargs["restore_checkpoint_path"] = args.load_checkpoint
        
        # Train policy
        make_inference, params, _ = train_ppo(env, eval_env, **train_kwargs)
        
        print(f"Training completed in {datetime.now() - start_time}")
        
        # Always render after training to see the final policy
        print("Rendering final trained policy...")
        command = None
        if args.env == "quadruped":
            command = jax.numpy.array([args.x_vel, args.y_vel, args.ang_vel])
        
        render_policy(eval_env, make_inference(params), command=command)
        
    else:
        # Load policy from checkpoint
        if not args.load_checkpoint:
            raise ValueError("Must provide --load_checkpoint when using --eval_only")
        
        from utils.checkpoint import load_params
        params = load_params(args.load_checkpoint)
        make_inference = make_inference_fn()
    
    # Render policy if requested
    if args.render or args.eval_only:
        print("Rendering policy...")
        command = None
        if args.env == "quadruped":
            command = jax.numpy.array([args.x_vel, args.y_vel, args.ang_vel])
        
        # Render with different commands to see versatility
        if args.env == "quadruped":
            print("Forward movement (x_vel=1.0, y_vel=0.0, ang_vel=0.0)")
            render_policy(eval_env, make_inference(params), 
                         command=jax.numpy.array([1.0, 0.0, 0.0]))
            
            print("Sideways movement (x_vel=0.0, y_vel=1.0, ang_vel=0.0)")
            render_policy(eval_env, make_inference(params), 
                         command=jax.numpy.array([0.0, 1.0, 0.0]))
            
            print("Turning (x_vel=0.5, y_vel=0.0, ang_vel=0.5)")
            render_policy(eval_env, make_inference(params), 
                         command=jax.numpy.array([0.5, 0.0, 0.5]))
        else:
            render_policy(eval_env, make_inference(params), command=command)


if __name__ == "__main__":
    main() 