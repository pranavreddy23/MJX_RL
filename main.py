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
from typing import Optional

# --- TensorBoard Import ---
from flax.metrics import tensorboard
from flax.training import orbax_utils
import orbax.checkpoint as ocp
from brax.io import model # Need model for param loading if using older checkpoints

# Need to import PPO train directly for eval_only mode setup
from brax.training.agents.ppo import train as ppo_brax_train
from brax.training.agents.ppo import networks as ppo_networks # Need network factory for eval
import functools # Need functools for partial

# Helper function to find latest numeric step checkpoint
def get_latest_step_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Finds the checkpoint with the highest step number."""
    latest_step = -1
    latest_ckpt_path = None
    if not os.path.isdir(checkpoint_dir):
        return None
    for item in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(path) and item.startswith('step_'):
            try:
                step = int(item.split('_')[-1])
                if step > latest_step:
                    latest_step = step
                    latest_ckpt_path = path
            except ValueError:
                continue
    return latest_ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(description="MJX RL Training")
    parser.add_argument("--env", type=str, default="quadruped", help="Environment name")
    parser.add_argument("--scene", type=str, default=None, help="Scene file (for quadruped)")
    parser.add_argument("--timesteps", type=int, default=100_000_000, help="Number of training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel environments")
    parser.add_argument("--num_evals", type=int, default=100, help="Number of evaluations during training (controls periodic checkpoint frequency)")
    parser.add_argument("--checkpoint_dir", type=str, default="/tmp/mjx_checkpoints", help="Directory to save checkpoints and logs")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Specific checkpoint path to load (for resuming training or specific eval)")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation, no training")
    parser.add_argument("--render", action="store_true", default=True, help="Render policy after training/evaluation") # Default to true for convenience
    parser.add_argument("--domain_rand", action="store_true", help="Use domain randomization during training")
    # parser.add_argument("--show_progress_plot", action="store_true", default=True, help="Show training progress plots") # Removed matplotlib plot flag

    # Quadruped specific args
    parser.add_argument("--x_vel", type=float, default=1.0, help="Target x velocity for quadruped")
    parser.add_argument("--y_vel", type=float, default=0.0, help="Target y velocity for quadruped")
    parser.add_argument("--ang_vel", type=float, default=0.0, help="Target angular velocity for quadruped")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(args.checkpoint_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))

    print(f"JAX devices: {jax.devices()}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    if not args.eval_only:
        print(f"TensorBoard log directory: {log_dir}")


    # --- Load Config ---
    # Needed for both training and eval (for network structure)
    config = get_ppo_config()
    config.num_timesteps = args.timesteps
    config.num_envs = args.num_envs
    config.num_evals = args.num_evals
    eval_frequency = config.num_timesteps // config.num_evals if config.num_evals > 0 else config.num_timesteps
    print(f"Evaluation frequency: approximately every {eval_frequency} steps")

    # Create environment
    env_kwargs = {}
    if args.env == "quadruped" and args.scene is not None:
        env_kwargs["scene_file"] = args.scene

    # Eval env is always needed
    eval_env = get_environment(args.env, **env_kwargs)


    # Training or loading policy
    if not args.eval_only:
        print(f"\n--- Starting Training ---")
        start_time = datetime.now()

        # --- Setup TensorBoard ---
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = tensorboard.SummaryWriter(log_dir)

        # Create training environment
        env = get_environment(args.env, **env_kwargs)

        train_kwargs = {
            "env": env,
            "eval_env": eval_env,
            "config": config,
            "seed": args.seed,
            "checkpoint_dir": args.checkpoint_dir,
            "summary_writer": summary_writer, # Pass the writer
        }

        if args.domain_rand:
            train_kwargs["use_domain_randomization"] = True

        # --- Automatic Resume Logic ---
        checkpoint_to_resume_from = None
        if args.load_checkpoint:
            # User explicitly specified a checkpoint to RESUME from
            checkpoint_to_resume_from = args.load_checkpoint
            print(f"Attempting to resume training from user-specified checkpoint: {checkpoint_to_resume_from}")
        else:
            # Check if best_checkpoint exists for automatic resuming
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_checkpoint')
            if os.path.exists(best_checkpoint_path):
                checkpoint_to_resume_from = best_checkpoint_path
                print(f"Found existing 'best_checkpoint'. Attempting to resume training from: {checkpoint_to_resume_from}")
            else:
                print("No 'best_checkpoint' found. Starting training from scratch.")

        if checkpoint_to_resume_from:
             if not os.path.exists(checkpoint_to_resume_from):
                 print(f"Warning: Checkpoint specified or found for resuming does not exist: {checkpoint_to_resume_from}. Starting from scratch.")
             else:
                 train_kwargs["restore_checkpoint_path"] = checkpoint_to_resume_from


        # --- Train policy ---
        make_inference, params, _ = train_ppo(**train_kwargs)

        print(f"--- Training completed in {datetime.now() - start_time} ---")

        # Create inference function from the *final* parameters returned by train_ppo
        # This is used if --render is True right after training finishes.
        inference_fn = make_inference(params)
        jit_inference_fn = jax.jit(inference_fn)
        print("Using final parameters from training run for rendering.")

    else: # --eval_only mode
        print(f"\n--- Evaluation Only Mode ---")
        # Load policy from checkpoint
        best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_checkpoint')
        final_model_path = os.path.join(args.checkpoint_dir, 'final_model')
        latest_step_path = get_latest_step_checkpoint(args.checkpoint_dir)

        checkpoint_to_load = None
        if args.load_checkpoint:
            # User explicitly specified a checkpoint
            checkpoint_to_load = args.load_checkpoint
            print(f"Loading user-specified checkpoint for evaluation: {checkpoint_to_load}")
        elif os.path.exists(best_checkpoint_path):
            checkpoint_to_load = best_checkpoint_path
            print(f"Loading 'best_checkpoint' for evaluation: {checkpoint_to_load}")
        elif latest_step_path:
            # Fallback to latest periodic step if best doesn't exist
            checkpoint_to_load = latest_step_path
            print(f"Loading latest step checkpoint for evaluation: {checkpoint_to_load}")
        elif os.path.exists(final_model_path):
             # Fallback to final model if best and periodic don't exist
             checkpoint_to_load = final_model_path
             print(f"Loading 'final_model' checkpoint for evaluation: {checkpoint_to_load}")
        else:
            raise FileNotFoundError(
                f"Cannot run eval_only. No checkpoint specified via --load_checkpoint, "
                f"and 'best_checkpoint', latest step, or 'final_model' not found in {args.checkpoint_dir}"
            )

        if not os.path.exists(checkpoint_to_load):
             raise FileNotFoundError(f"Specified or inferred checkpoint not found for evaluation: {checkpoint_to_load}")

        # --- Load the parameters ---
        print(f"Loading parameters from: {checkpoint_to_load}")
        # Use Orbax checkpointer directly for loading, as it's used for saving
        checkpointer = ocp.PyTreeCheckpointer()
        # We need a target structure for Orbax to restore into. Get it from the network factory.
        print("Setting up network structure for parameter loading...")
        make_networks_factory_eval = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=config.network.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=config.network.value_hidden_layer_sizes
        )
        make_inference_eval, params_structure, _ = ppo_brax_train(
            environment=eval_env,
            num_timesteps=0,
            network_factory=make_networks_factory_eval
        )
        params = checkpointer.restore(checkpoint_to_load, args=ocp.args.StandardRestore(params_structure))
        print("Parameters loaded successfully.")


        # Create the actual inference function using the loaded parameters
        inference_fn = make_inference_eval(params)
        jit_inference_fn = jax.jit(inference_fn)
        print("Inference function ready.")


    # Render policy if requested (default is now true)
    if args.render:
        print("\n--- Rendering Policy ---")
        command = None
        if args.env == "quadruped":
            command = jax.numpy.array([args.x_vel, args.y_vel, args.ang_vel])

        # Prepare for visualization
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)

        # Initialize the environment
        rng = jax.random.PRNGKey(args.seed + 1) # Use different seed for eval
        state = jit_reset(rng)

        # Set command for quadruped
        if args.env == "quadruped" and command is not None and hasattr(state, 'info') and 'command' in state.info:
            state.info['command'] = command
            print(f"Setting command: x_vel={args.x_vel}, y_vel={args.y_vel}, ang_vel={args.ang_vel}")

        # Collect trajectory
        rollout = [state.pipeline_state]
        rewards = []

        n_steps = config.episode_length # Use episode length from config
        render_start_time = time.time()
        for i in range(n_steps):
            act_rng, rng = jax.random.split(rng)
            action, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, action)
            rollout.append(state.pipeline_state)
            rewards.append(float(state.reward))
            if state.done:
                print(f"Episode finished early at step {i+1}")
                break
        render_end_time = time.time()
        print(f"Rollout generation time: {render_end_time - render_start_time:.2f}s")


        # Print statistics
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        total_reward = sum(rewards)
        print(f"Average reward over {len(rewards)} steps: {avg_reward:.3f}")
        print(f"Total reward: {total_reward:.3f}")

        # Render the trajectory
        render_every = 2 # Keep this fixed or make it an arg
        render_fps = 1.0 / eval_env.dt / render_every

        try:
            print("Rendering video...")
            frames = eval_env.render(rollout[::render_every], camera='track')

            # Save video
            video_path = os.path.join(args.checkpoint_dir, 'policy_render.mp4')
            media.write_video(video_path, frames, fps=render_fps)
            print(f"Video saved to {video_path}")

            # Try to display video if possible (useful in notebooks)
            # try:
            #     media.show_video(frames, fps=render_fps)
            # except Exception as e:
            #     print(f"Could not display video inline: {e}")
            #     pass # Don't crash if display fails
        except Exception as e:
            print(f"Error during video rendering: {e}")
            print("This might be due to missing ffmpeg or other rendering issues.")

        # Also try to save HTML visualization
        try:
            print("Generating HTML visualization...")
            # Ensure system dt matches eval_env dt for html rendering
            html_sys = eval_env.sys.tree_replace({'opt.timestep': eval_env.dt})
            html_str = html.render(html_sys, rollout)
            html_path = os.path.join(args.checkpoint_dir, 'policy_visualization.html')
            with open(html_path, 'w') as f:
                f.write(html_str)
            print(f"HTML visualization saved to {html_path}")
        except Exception as e:
            print(f"Error creating HTML visualization: {e}")


if __name__ == "__main__":
    main() 