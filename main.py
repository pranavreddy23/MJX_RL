"""
Main entry point for MJX RL training.
This file provides a high-level interface for training and evaluating policies.
"""

import os
import argparse
import jax
import time
from datetime import datetime
import mediapy as media
from brax.io import html
import traceback  # Add explicit import for traceback

from environments import get_environment
from policies import train_ppo, train_humanoid  # Import both training functions
from utils.rendering import render_policy
from configs.default_configs import get_ppo_config
from typing import Optional, List, Dict, Any, Tuple

# --- TensorBoard Import ---
from flax.metrics import tensorboard
from flax.training import orbax_utils
import orbax.checkpoint as ocp
from brax.io import model

# Need imports for policy loading
from brax.training.agents.ppo import train as ppo_brax_train
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import checkpoint as ppo_checkpoint_loader
import functools

# --- Checkpoint Finding Function (Remains the same) ---
def find_brax_checkpoint(brax_log_dir: str) -> Optional[str]:
    """
    Finds the latest valid Brax PPO checkpoint directory within the log directory.
    Prioritizes 'final_model' if it exists and points to a valid checkpoint.
    """
    latest_valid_step = -1
    latest_valid_ckpt_path = None
    print(f"Scanning for Brax checkpoints in: {brax_log_dir}")

    if not os.path.isdir(brax_log_dir):
        print(f"Warning: Brax log directory '{brax_log_dir}' not found.")
        return None

    # Scan direct subdirectories of brax_log_dir for step-numbered checkpoints
    for item in os.listdir(brax_log_dir):
        path = os.path.join(brax_log_dir, item)
        # Check if it's a directory and looks like a step number
        if os.path.isdir(path) and item.isdigit():
            try:
                step = int(item)
                # Check for the necessary files created by ppo_checkpoint.save
                metadata_path = os.path.join(path, '_METADATA')
                config_path = os.path.join(path, 'ppo_network_config.json')

                if os.path.exists(metadata_path) and os.path.exists(config_path):
                    if step > latest_valid_step:
                        latest_valid_step = step
                        latest_valid_ckpt_path = path
                        print(f"  Found valid step checkpoint: {path} (Step: {step})")
            except ValueError:
                continue # Ignore non-numeric directory names

    # Now check for 'final_model' and prioritize it if valid
    final_model_path = os.path.join(brax_log_dir, 'final_model')
    if os.path.exists(final_model_path):
        print(f"Found potential 'final_model' at {final_model_path}")
        real_final_model_path = os.path.realpath(final_model_path)
        if os.path.isdir(real_final_model_path):
            metadata_path = os.path.join(real_final_model_path, '_METADATA')
            config_path = os.path.join(real_final_model_path, 'ppo_network_config.json')

            if os.path.exists(metadata_path) and os.path.exists(config_path):
                print(f"Using 'final_model' pointing to valid checkpoint directory: {real_final_model_path}")
                return real_final_model_path # Return final_model path if valid
            else:
                print(f"Ignoring 'final_model' link/dir {final_model_path} as target {real_final_model_path} is missing required checkpoint files.")
        else:
            print(f"Ignoring 'final_model' at {final_model_path} as it's not a valid directory or link to one.")

    # If final_model wasn't found/valid, return the latest step checkpoint found (if any)
    if latest_valid_ckpt_path:
        print(f"Using latest valid step checkpoint found: {latest_valid_ckpt_path} (Step: {latest_valid_step})")
    else:
        print(f"No valid step-numbered Brax checkpoints found in {brax_log_dir}.")

    return latest_valid_ckpt_path

# --- Command Line Arguments (Remains the same) ---
def parse_args():
    parser = argparse.ArgumentParser(description="MJX RL Training with Brax Checkpointing")
    # Directory settings
    parser.add_argument("--brax_checkpoint_dir", type=str, default="./checkpoints",
                       help="Base directory for Brax logs and checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="./old_checkpoints",
                       help="[DEPRECATED/ORBAX ONLY] Directory for old Orbax checkpoints/logs")

    # Environment settings
    parser.add_argument("--env", type=str, default="quadruped",
                       help="Environment name (quadruped or humanoid)")
    parser.add_argument("--scene", type=str, default=None,
                       help="Scene file (for quadruped)")

    # Training settings
    parser.add_argument("--timesteps", type=int, default=20_000_000,
                       help="Number of training timesteps")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--num_envs", type=int, default=2048,
                       help="Number of parallel environments")
    parser.add_argument("--num_evals", type=int, default=100,
                       help="Number of evaluations during training")
    parser.add_argument("--domain_rand", action="store_true",
                       help="Use domain randomization during training")

    # Checkpoint and evaluation
    parser.add_argument("--load_checkpoint", type=str, default=None,
                       help="Specific Brax checkpoint path to load")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only run evaluation, no training")
    parser.add_argument("--render", action="store_true", default=True,
                       help="Render policy after training/evaluation")

    # Quadruped-specific settings
    parser.add_argument("--x_vel", type=float, default=1.0,
                       help="Target x velocity for quadruped")
    parser.add_argument("--y_vel", type=float, default=0.0,
                       help="Target y velocity for quadruped")
    parser.add_argument("--ang_vel", type=float, default=0.0,
                       help="Target angular velocity for quadruped")

    return parser.parse_args()

# --- Main Function ---
def main():
    args = parse_args()
    env_name = args.env.lower()

    # Validate environment name
    if env_name not in ["quadruped", "humanoid"]:
        raise ValueError(f"Unsupported environment: {env_name}. Choose 'quadruped' or 'humanoid'.")

    print(f"JAX devices: {jax.devices()}")

    # --- Environment-Specific Setup ---
    # Checkpoint directory
    brax_log_dir = os.path.abspath(os.path.join(args.brax_checkpoint_dir, env_name))
    os.makedirs(brax_log_dir, exist_ok=True)
    print(f"Using checkpoint directory: {brax_log_dir}")

    # PPO Config
    config = get_ppo_config()
    config.num_timesteps = args.timesteps
    config.num_envs = args.num_envs
    config.num_evals = args.num_evals
    eval_frequency = config.num_timesteps // config.num_evals if config.num_evals > 0 else config.num_timesteps
    print(f"Evaluation frequency (controls internal checkpointing): approximately every {eval_frequency} steps")

    # Environment kwargs
    env_kwargs = {}
    if env_name == "quadruped" and args.scene is not None:
        env_kwargs["scene_file"] = args.scene

    # Create eval environment (needed for both train and eval modes)
    eval_env = get_environment(env_name, **env_kwargs)

    # Create environment-specific network factory for evaluation/loading
    if env_name == 'humanoid':
        make_networks_factory_eval = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(256, 256, 256),
            value_hidden_layer_sizes=(256, 256, 256)
        )
    else: # quadruped
        make_networks_factory_eval = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=config.network.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=config.network.value_hidden_layer_sizes
        )

    # --- Training or Evaluation ---
    jit_inference_fn = None

    if not args.eval_only:
        # --- Training Mode ---
        print(f"\n--- Starting {env_name} Training ---")
        start_time = datetime.now()

        # Setup TensorBoard
        tensorboard_log_dir = os.path.join(brax_log_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        summary_writer = tensorboard.SummaryWriter(tensorboard_log_dir)
        print(f"TensorBoard log directory: {tensorboard_log_dir}")

        # Create training environment
        train_env = get_environment(env_name, **env_kwargs)

        # Find checkpoint to resume from
        checkpoint_to_resume_from = None
        if args.load_checkpoint:
            if os.path.exists(args.load_checkpoint):
                 checkpoint_to_resume_from = os.path.abspath(args.load_checkpoint)
                 print(f"Attempting to resume from user-specified checkpoint: {checkpoint_to_resume_from}")
            else:
                 print(f"Warning: User specified checkpoint '{args.load_checkpoint}' not found. Starting fresh.")
        else:
            checkpoint_to_resume_from = find_brax_checkpoint(brax_log_dir)
            if checkpoint_to_resume_from:
                print(f"Found checkpoint to resume from: {checkpoint_to_resume_from}")
            else:
                print("No suitable checkpoint found. Starting training from scratch.")

        # Prepare training arguments
        train_kwargs = {
            "env": train_env,
            "eval_env": eval_env,
            "seed": args.seed,
            "brax_log_dir": brax_log_dir, # Pass environment-specific directory
            "summary_writer": summary_writer,
        }
        if checkpoint_to_resume_from and os.path.exists(checkpoint_to_resume_from):
            train_kwargs["restore_checkpoint_path"] = checkpoint_to_resume_from
            print(f"Will restore training from: {checkpoint_to_resume_from}")

        # Call appropriate training function
        if env_name == "humanoid":
            print("Using specialized humanoid training function")
            make_inference, params, _ = train_humanoid(**train_kwargs)
        else: # quadruped
            print("Using standard PPO training function")
            train_kwargs["config"] = config # Add config for standard training
            if args.domain_rand:
                train_kwargs["use_domain_randomization"] = True
                print("Domain randomization enabled for quadruped.")
            make_inference, params, _ = train_ppo(**train_kwargs)

        print(f"--- Training completed in {datetime.now() - start_time} ---")

        # Get inference function from training results
        inference_fn = make_inference(params)
        jit_inference_fn = jax.jit(inference_fn)
        print("Using final parameters from training run for rendering.")

    else:
        # --- Evaluation Only Mode ---
        print(f"\n--- Evaluation Only Mode for {env_name} ---")

        # Find checkpoint for evaluation
        checkpoint_to_load = None
        if args.load_checkpoint:
            if os.path.exists(args.load_checkpoint):
                 checkpoint_to_load = os.path.abspath(args.load_checkpoint)
                 print(f"Loading user-specified checkpoint: {checkpoint_to_load}")
            else:
                 raise FileNotFoundError(f"User specified checkpoint '{args.load_checkpoint}' not found.")
        else:
            checkpoint_to_load = find_brax_checkpoint(brax_log_dir)
            if not checkpoint_to_load:
                 raise FileNotFoundError(f"Cannot run eval_only. No checkpoint found in {brax_log_dir}")
            print(f"Loading latest checkpoint for evaluation: {checkpoint_to_load}")

        # Load the policy
        try:
            print(f"Loading policy using Brax loader from: {checkpoint_to_load}")
            # Use the eval network factory created earlier
            inference_fn = ppo_checkpoint_loader.load_policy(
                path=checkpoint_to_load,
                network_factory=make_networks_factory_eval,
                deterministic=True
            )
            jit_inference_fn = jax.jit(inference_fn)
            print("Policy loaded and inference function ready.")
        except Exception as e:
             print(f"Error loading policy with Brax loader: {e}")
             traceback.print_exc()
             raise

    # --- Rendering Logic (Executed if training OR eval_only finished) ---
    if jit_inference_fn is None:
        print("Skipping rendering because no inference function was created (likely an error).")
        return

    if args.render:
        print(f"\n--- Rendering {env_name} Policy ---")
        camera = 'side' if env_name == 'humanoid' else 'track'

        command = None
        if env_name == "quadruped":
            command = jax.numpy.array([args.x_vel, args.y_vel, args.ang_vel])

        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        rng = jax.random.PRNGKey(args.seed + 1)
        state = jit_reset(rng)

        if env_name == "quadruped" and command is not None and hasattr(state, 'info') and 'command' in state.info:
            state.info['command'] = command
            print(f"Setting command: x_vel={args.x_vel}, y_vel={args.y_vel}, ang_vel={args.ang_vel}")

        rollout = [state.pipeline_state]
        rewards = []
        n_steps = config.episode_length # Use episode length from config
        render_start_time = time.time()
        print(f"Generating rollout for {n_steps} steps...")
        for i in range(n_steps):
            act_rng, rng = jax.random.split(rng)
            try:
                action, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, action)
                rollout.append(state.pipeline_state)
                rewards.append(float(state.reward))
                if state.done:
                    print(f"Episode finished early at step {i+1}")
                    break
            except Exception as e:
                 print(f"\nError during rollout step {i}: {e}")
                 traceback.print_exc()
                 break
        render_end_time = time.time()
        print(f"Rollout generation time: {render_end_time - render_start_time:.2f}s")

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        total_reward = sum(rewards)
        print(f"Average reward over {len(rewards)} steps: {avg_reward:.3f}")
        print(f"Total reward: {total_reward:.3f}")

        # Save rendering outputs relative to the env-specific log dir
        render_output_dir = brax_log_dir
        render_every = 2
        render_fps = 1.0 / eval_env.dt / render_every

        # Render video
        try:
            print(f"Rendering video with '{camera}' camera...")
            frames = eval_env.render(rollout[::render_every], camera=camera)
            video_path = os.path.join(render_output_dir, 'policy_render.mp4')
            media.write_video(video_path, frames, fps=render_fps)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Error during video rendering: {e}")
            traceback.print_exc()

        # Generate HTML visualization
        try:
            print("Generating HTML visualization...")
            html_sys = eval_env.sys.tree_replace({'opt.timestep': eval_env.dt})
            html_str = html.render(html_sys, rollout)
            html_path = os.path.join(render_output_dir, 'policy_visualization.html')
            with open(html_path, 'w') as f:
                f.write(html_str)
            print(f"HTML visualization saved to {html_path}")
        except Exception as e:
            print(f"Error creating HTML visualization: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()