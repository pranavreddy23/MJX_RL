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
# from IPython.display import HTML # Removed IPython dependency

from environments import get_environment
from policies import train_ppo
from utils.rendering import render_policy
from configs.default_configs import get_ppo_config
from typing import Optional, List

# --- TensorBoard Import ---
from flax.metrics import tensorboard
from flax.training import orbax_utils
import orbax.checkpoint as ocp
from brax.io import model # Need model for param loading if using older checkpoints

# Need to import PPO train directly for eval_only mode setup
from brax.training.agents.ppo import train as ppo_brax_train
from brax.training.agents.ppo import networks as ppo_networks # Need network factory for eval
import functools # Need functools for partial

# Need imports for eval_only loading using Brax loader
from brax.training.agents.ppo import checkpoint as ppo_checkpoint_loader
from brax.training.agents.ppo import networks as ppo_networks # Still need network factory

# --- Updated Checkpoint Finding for Brax Format ---
def find_brax_checkpoint(brax_log_dir: str) -> Optional[str]:
    """
    Finds the latest valid Brax PPO checkpoint directory within the log directory.
    Checks for step-numbered directories containing '_METADATA' and 'ppo_network_config.json'.
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
                # Use the correct config filename
                config_path = os.path.join(path, 'ppo_network_config.json')

                if os.path.exists(metadata_path) and os.path.exists(config_path):
                    if step > latest_valid_step:
                        latest_valid_step = step
                        latest_valid_ckpt_path = path
                        print(f"  Found valid step checkpoint: {path} (Step: {step})")
                # else:
                #      print(f"  Skipping potential ckpt {path}: missing _METADATA or ppo_network_config.json")

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
# --- End Checkpoint Finding ---


def parse_args():
    parser = argparse.ArgumentParser(description="MJX RL Training with Brax Checkpointing")
    # Add new argument for Brax checkpoints
    parser.add_argument("--brax_checkpoint_dir", type=str, default="/tmp/mjx_brax_checkpoints", help="Base directory for Brax logs and checkpoints")
    # Keep old arg but maybe mark as deprecated or for Orbax format only
    parser.add_argument("--checkpoint_dir", type=str, default="/tmp/mjx_checkpoints", help="[DEPRECATED/ORBAX ONLY] Directory for old Orbax checkpoints/logs")
    parser.add_argument("--env", type=str, default="quadruped", help="Environment name")
    parser.add_argument("--scene", type=str, default=None, help="Scene file (for quadruped)")
    parser.add_argument("--timesteps", type=int, default=20_000_000, help="Number of training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel environments")
    parser.add_argument("--num_evals", type=int, default=100, help="Number of evaluations during training (controls periodic checkpoint frequency)")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Specific Brax checkpoint path to load (e.g., .../checkpoints/000...)")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation, no training")
    parser.add_argument("--render", action="store_true", default=True, help="Render policy after training/evaluation")
    parser.add_argument("--domain_rand", action="store_true", help="Use domain randomization during training")

    parser.add_argument("--x_vel", type=float, default=1.0, help="Target x velocity for quadruped")
    parser.add_argument("--y_vel", type=float, default=0.0, help="Target y velocity for quadruped")
    parser.add_argument("--ang_vel", type=float, default=0.0, help="Target angular velocity for quadruped")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Use the NEW directory for Brax logs and checkpoints ---
    brax_log_dir = args.brax_checkpoint_dir
    os.makedirs(brax_log_dir, exist_ok=True)
    # Keep TensorBoard logs within this new structure
    tensorboard_log_dir = os.path.join(brax_log_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))

    print(f"JAX devices: {jax.devices()}")
    print(f"Brax log/checkpoint directory: {brax_log_dir}")
    if not args.eval_only:
        print(f"TensorBoard log directory: {tensorboard_log_dir}")

    config = get_ppo_config()
    config.num_timesteps = args.timesteps
    config.num_envs = args.num_envs
    config.num_evals = args.num_evals
    eval_frequency = config.num_timesteps // config.num_evals if config.num_evals > 0 else config.num_timesteps
    print(f"Evaluation frequency (controls internal checkpointing): approximately every {eval_frequency} steps")

    env_kwargs = {}
    if args.env == "quadruped" and args.scene is not None:
        env_kwargs["scene_file"] = args.scene
    # Eval env needed for eval_only mode structure setup
    eval_env = get_environment(args.env, **env_kwargs)

    if not args.eval_only:
        print(f"\n--- Starting Training ---")
        start_time = datetime.now()
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        summary_writer = tensorboard.SummaryWriter(tensorboard_log_dir)
        # Training env
        env = get_environment(args.env, **env_kwargs)

        train_kwargs = {
            "env": env,
            "eval_env": eval_env,
            "config": config,
            "seed": args.seed,
            "brax_log_dir": brax_log_dir, # Pass the new log dir to train_ppo
            "summary_writer": summary_writer,
        }
        if args.domain_rand:
            train_kwargs["use_domain_randomization"] = True

        # --- Automatic Resume Logic (using find_brax_checkpoint on NEW dir) ---
        checkpoint_to_resume_from = None
        if args.load_checkpoint:
            # User explicitly specified a Brax checkpoint path
            if os.path.exists(args.load_checkpoint):
                 checkpoint_to_resume_from = args.load_checkpoint
                 print(f"Attempting to resume training from user-specified checkpoint: {checkpoint_to_resume_from}")
            else:
                 print(f"Warning: User specified checkpoint '{args.load_checkpoint}' not found. Starting fresh.")
        else:
            # Find the latest Brax checkpoint in the new directory
            checkpoint_to_resume_from = find_brax_checkpoint(brax_log_dir)
            if checkpoint_to_resume_from:
                print(f"Found checkpoint to resume from: {checkpoint_to_resume_from}")
            else:
                print("No suitable checkpoint found in Brax structure. Starting training from scratch.")

        # Pass the path to train_ppo, which passes it to ppo.train's restore_checkpoint_path
        if checkpoint_to_resume_from and os.path.exists(checkpoint_to_resume_from):
             train_kwargs["restore_checkpoint_path"] = checkpoint_to_resume_from
        # --- End Resume Logic ---


        # --- Train policy ---
        make_inference, params, _ = train_ppo(**train_kwargs)
        print(f"--- Training completed in {datetime.now() - start_time} ---")

        # The 'params' returned are likely just the network weights, suitable for inference
        inference_fn = make_inference(params)
        jit_inference_fn = jax.jit(inference_fn)
        print("Using final parameters from training run for rendering.")

    else: # --eval_only mode
        print(f"\n--- Evaluation Only Mode ---")

        # --- Find Checkpoint for Eval (in NEW dir) ---
        checkpoint_to_load = None
        if args.load_checkpoint:
             # User explicitly specified a Brax checkpoint path
            if os.path.exists(args.load_checkpoint):
                 checkpoint_to_load = args.load_checkpoint
                 print(f"Loading user-specified checkpoint for evaluation: {checkpoint_to_load}")
            else:
                 raise FileNotFoundError(f"User specified checkpoint '{args.load_checkpoint}' not found.")
        else:
            # Find the latest Brax checkpoint in the new directory
            checkpoint_to_load = find_brax_checkpoint(brax_log_dir)
            if not checkpoint_to_load:
                 raise FileNotFoundError(f"Cannot run eval_only. No checkpoint specified and no valid Brax checkpoint found in {brax_log_dir}")
            print(f"Loading checkpoint for evaluation: {checkpoint_to_load}")

        if not checkpoint_to_load or not os.path.exists(checkpoint_to_load):
             raise FileNotFoundError(f"Specified or inferred checkpoint not found for evaluation: {checkpoint_to_load}")
        # --- End Find Checkpoint ---


        # --- Load the policy using Brax PPO loader ---
        print(f"Loading policy using Brax loader from: {checkpoint_to_load}")
        # Network factory structure is needed by the loader
        make_networks_factory_eval = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=config.network.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=config.network.value_hidden_layer_sizes
        )
        try:
            # Pass the specific checkpoint dir (e.g., .../checkpoints/000... or .../final_model)
            inference_fn = ppo_checkpoint_loader.load_policy(
                path=checkpoint_to_load,
                network_factory=make_networks_factory_eval,
                deterministic=True
            )
            jit_inference_fn = jax.jit(inference_fn)
            print("Policy loaded and inference function ready.")
        except Exception as e:
             print(f"Error loading policy with Brax loader: {e}")
             print("Ensure the checkpoint path is correct and contains the necessary files (like config.json).")
             import traceback
             traceback.print_exc()
             raise
        # --- End Policy Loading ---

    # --- Rendering Logic ---
    if args.render:
        print("\n--- Rendering Policy ---")
        command = None
        if args.env == "quadruped":
            command = jax.numpy.array([args.x_vel, args.y_vel, args.ang_vel])

        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        rng = jax.random.PRNGKey(args.seed + 1)
        state = jit_reset(rng)

        if args.env == "quadruped" and command is not None and hasattr(state, 'info') and 'command' in state.info:
            state.info['command'] = command # NOTE: state.info might be immutable in some JAX versions
            print(f"Setting command: x_vel={args.x_vel}, y_vel={args.y_vel}, ang_vel={args.ang_vel}")

        rollout = [state.pipeline_state]
        rewards = []
        n_steps = config.episode_length
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
                 break # Stop rollout if inference fails
        render_end_time = time.time()
        print(f"Rollout generation time: {render_end_time - render_start_time:.2f}s")

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        total_reward = sum(rewards)
        print(f"Average reward over {len(rewards)} steps: {avg_reward:.3f}")
        print(f"Total reward: {total_reward:.3f}")

        # Save rendering outputs relative to the NEW log dir
        render_output_dir = brax_log_dir
        render_every = 2
        render_fps = 1.0 / eval_env.dt / render_every

        try:
            print("Rendering video...")
            frames = eval_env.render(rollout[::render_every], camera='track')
            video_path = os.path.join(render_output_dir, 'policy_render.mp4') # Save in new dir
            media.write_video(video_path, frames, fps=render_fps)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Error during video rendering: {e}")

        try:
            print("Generating HTML visualization...")
            html_sys = eval_env.sys.tree_replace({'opt.timestep': eval_env.dt})
            html_str = html.render(html_sys, rollout)
            html_path = os.path.join(render_output_dir, 'policy_visualization.html') # Save in new dir
            with open(html_path, 'w') as f:
                f.write(html_str)
            print(f"HTML visualization saved to {html_path}")
        except Exception as e:
            print(f"Error creating HTML visualization: {e}")


if __name__ == "__main__":
    main() 