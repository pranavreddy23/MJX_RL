"""
PPO policy implementation for MJX environments using Brax internal checkpointing.
"""

import functools
import os
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jp
from flax.metrics import tensorboard
from datetime import datetime
import shutil

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.envs.base import Env
from ml_collections import ConfigDict
from brax.training.agents.ppo import checkpoint as ppo_checkpoint


def train_ppo(
    env: Env,
    eval_env: Optional[Env] = None,
    config: Optional[ConfigDict] = None,
    seed: int = 0,
    brax_log_dir: str = "./checkpoints/quadruped", # Default to quadruped subdir
    restore_checkpoint_path: Optional[str] = None,
    use_domain_randomization: bool = False,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    summary_writer: Optional[tensorboard.SummaryWriter] = None,
) -> Tuple[Callable, Any, Dict[str, Any]]:
    """
    Train a policy using PPO, relying on Brax internal checkpointing,
    saving checkpoints explicitly after training.
    """
    # Load default config if none provided
    if config is None:
        from configs.default_configs import get_ppo_config
        config = get_ppo_config()

    # Ensure Brax log directory exists and use absolute path
    brax_log_dir = os.path.abspath(brax_log_dir)
    os.makedirs(brax_log_dir, exist_ok=True)
    print(f"Checkpoint log directory (absolute): {brax_log_dir}")

    # Default progress function (unchanged)
    if progress_fn is None:
        times = [datetime.now()]
        print("Using default progress function for console/TensorBoard logging.")
        def default_progress(step, metrics):
            if step > 0 and len(times) == 1: times.append(datetime.now()); print(f"First step time (incl. JIT): {times[-1] - times[0]}")
            elif step > 0: times.append(datetime.now())
            if summary_writer:
                for key, value in metrics.items():
                    if jp.isscalar(value) and jp.isfinite(value):
                        try: summary_writer.scalar(key, value, step)
                        except Exception as e: print(f"Warning: Could not log metric '{key}' to TensorBoard: {e}")
            current_eval_reward = metrics.get('eval/episode_reward', jp.nan)
            current_eval_reward_std = metrics.get('eval/episode_reward_std', 0)
            print(f"\n--- Step: {step}/{config.num_timesteps} ---")
            print(f"Eval Reward: {current_eval_reward:.3f} +/- {current_eval_reward_std:.3f}")
            if len(times) > 1: print(f"Total training time: {(times[-1] - times[1]).total_seconds()/60:.2f} minutes")
        progress_fn = default_progress


    # Network factory (unchanged)
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.network.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=config.network.value_hidden_layer_sizes
    )

    # Domain randomization import (unchanged)
    from utils.domain_rand import domain_randomize

    # Configure PPO training (no internal checkpointing)
    train_fn_partial = functools.partial(
        ppo.train,
        num_timesteps=config.num_timesteps,
        num_evals=config.num_evals,
        reward_scaling=config.reward_scaling,
        episode_length=config.episode_length,
        normalize_observations=config.normalize_observations,
        action_repeat=config.action_repeat,
        unroll_length=config.unroll_length,
        num_minibatches=config.num_minibatches,
        num_updates_per_batch=config.num_updates_per_batch,
        discounting=config.discounting,
        learning_rate=config.learning_rate,
        entropy_cost=config.entropy_cost,
        num_envs=config.num_envs,
        batch_size=config.batch_size,
        network_factory=make_networks_factory,
        seed=seed,
    )

    # Add domain randomization if requested
    if use_domain_randomization:
        train_fn_partial = functools.partial(train_fn_partial, randomization_fn=domain_randomize)

    # Prepare training arguments
    train_kwargs = {
        "environment": env,
        "progress_fn": progress_fn,
        "eval_env": eval_env,
    }
    if restore_checkpoint_path is not None:
        restore_checkpoint_path = os.path.abspath(restore_checkpoint_path)
        print(f"Passing restore_checkpoint_path to ppo.train: {restore_checkpoint_path}")
        train_kwargs["restore_checkpoint_path"] = restore_checkpoint_path
    else:
        print("Starting fresh training (no restore_checkpoint_path provided)")

    # Train the policy
    make_inference_fn, params, training_metrics = train_fn_partial(**train_kwargs)

    # --- Simplified Explicit Save Checkpoint AFTER Training ---
    if brax_log_dir:
        step = config.num_timesteps
        network_config = ppo_checkpoint.network_config(
            observation_size=env.observation_size,
            action_size=env.action_size,
            normalize_observations=config.normalize_observations,
            network_factory=make_networks_factory
        )
        expected_checkpoint_path = os.path.join(brax_log_dir, f'{step:012d}')

        try:
            print(f"\n--- Saving final checkpoint for step {step} ---")
            print(f"Target directory: {expected_checkpoint_path}")
            # Call save, ignore its None return value
            ppo_checkpoint.save(
                path=brax_log_dir,
                step=step,
                params=params,
                config=network_config
            )
            print(f"ppo_checkpoint.save call completed.")

            # --- Simplified Symlink Creation ---
            final_model_path = os.path.join(brax_log_dir, "final_model")
            print(f"Attempting to create/update 'final_model' symlink -> {expected_checkpoint_path}")

            # Force remove existing final_model (link, file, or dir)
            if os.path.lexists(final_model_path): # Use lexists to detect broken links too
                if os.path.islink(final_model_path):
                    print(f"Removing existing symlink: {final_model_path}")
                    os.unlink(final_model_path)
                elif os.path.isdir(final_model_path):
                     print(f"Removing existing directory: {final_model_path}")
                     shutil.rmtree(final_model_path)
                else:
                     print(f"Removing existing file: {final_model_path}")
                     os.remove(final_model_path)

            # Create the new symlink (only on non-Windows)
            if os.name != 'nt':
                 # Verify source directory exists before creating link
                 if os.path.isdir(expected_checkpoint_path):
                      os.symlink(expected_checkpoint_path, final_model_path, target_is_directory=True)
                      print(f"Successfully created 'final_model' symlink.")
                 else:
                      print(f"!!! Source checkpoint directory NOT FOUND: {expected_checkpoint_path}. Cannot create symlink.")
            else: # Basic copy for Windows
                 if os.path.isdir(expected_checkpoint_path):
                      shutil.copytree(expected_checkpoint_path, final_model_path)
                      print(f"Successfully created 'final_model' copy (Windows).")
                 else:
                      print(f"!!! Source checkpoint directory NOT FOUND: {expected_checkpoint_path}. Cannot create copy.")
            # --- End Simplified Symlink Creation ---

        except Exception as e:
            print(f"!!! ERROR during final checkpoint saving or symlink creation: {e}")
            import traceback
            traceback.print_exc()

    if summary_writer:
        summary_writer.close()

    return make_inference_fn, params, training_metrics

def train_humanoid(
    env: Env,
    eval_env: Optional[Env] = None,
    brax_log_dir: str = "./checkpoints/humanoid",
    restore_checkpoint_path: Optional[str] = None,
    seed: int = 0,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    summary_writer: Optional[tensorboard.SummaryWriter] = None,
) -> Tuple[Callable, Any, Dict[str, Any]]:
    """
    Train a humanoid policy using PPO with humanoid-specific hyperparameters,
    saving checkpoints explicitly after training, mirroring train_ppo structure.
    """
    # Convert to absolute path if not already
    brax_log_dir = os.path.abspath(brax_log_dir)
    print(f"Humanoid checkpoint directory (absolute): {brax_log_dir}")
    os.makedirs(brax_log_dir, exist_ok=True)

    # Default progress function (unchanged)
    if progress_fn is None:
        times = [datetime.now()]
        print("Using default progress function for console/TensorBoard logging.")
        def default_progress(step, metrics):
            if step > 0 and len(times) == 1: times.append(datetime.now()); print(f"First step time (incl. JIT): {times[-1] - times[0]}")
            elif step > 0: times.append(datetime.now())
            if summary_writer:
                for key, value in metrics.items():
                    if jp.isscalar(value) and jp.isfinite(value):
                        try: summary_writer.scalar(key, value, step)
                        except Exception as e: print(f"Warning: Could not log metric '{key}' to TensorBoard: {e}")
            current_eval_reward = metrics.get('eval/episode_reward', jp.nan)
            current_eval_reward_std = metrics.get('eval/episode_reward_std', 0)
            print(f"\n--- Step: {step}/20000000 ---")
            print(f"Eval Reward: {current_eval_reward:.3f} +/- {current_eval_reward_std:.3f}")
            if len(times) > 1: print(f"Total training time: {(times[-1] - times[1]).total_seconds()/60:.2f} minutes")
        progress_fn = default_progress

    # Network factory (unchanged)
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 256, 256),
        value_hidden_layer_sizes=(256, 256, 256)
    )

    # --- Configure PPO Training (No internal checkpointing) ---
    train_fn_partial = functools.partial(
        ppo.train,
        num_timesteps=20_000_000,
        num_evals=5,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=24,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=3072,
        batch_size=512,
        network_factory=make_networks_factory,
        seed=seed,
    )

    # Prepare training arguments (unchanged)
    train_kwargs = {
        "environment": env,
        "progress_fn": progress_fn,
        "eval_env": eval_env,
    }
    if restore_checkpoint_path is not None:
        restore_checkpoint_path = os.path.abspath(restore_checkpoint_path)
        print(f"Passing restore_checkpoint_path to ppo.train: {restore_checkpoint_path}")
        train_kwargs["restore_checkpoint_path"] = restore_checkpoint_path
    else:
        print("Starting fresh training (no restore_checkpoint_path provided)")

    # Train the policy (unchanged)
    make_inference_fn, params, training_metrics = train_fn_partial(**train_kwargs)

    # --- Simplified Explicit Save Checkpoint AFTER Training ---
    if brax_log_dir:
        step = 20_000_000
        network_config = ppo_checkpoint.network_config(
            observation_size=env.observation_size,
            action_size=env.action_size,
            normalize_observations=True,
            network_factory=make_networks_factory
        )
        expected_checkpoint_path = os.path.join(brax_log_dir, f'{step:012d}')

        try:
            print(f"\n--- Saving final checkpoint for step {step} ---")
            print(f"Target directory: {expected_checkpoint_path}")
            # Call save, ignore its None return value
            ppo_checkpoint.save(
                path=brax_log_dir,
                step=step,
                params=params,
                config=network_config
            )
            print(f"ppo_checkpoint.save call completed.")

            # --- Simplified Symlink Creation ---
            final_model_path = os.path.join(brax_log_dir, "final_model")
            print(f"Attempting to create/update 'final_model' symlink -> {expected_checkpoint_path}")

            # Force remove existing final_model (link, file, or dir)
            if os.path.lexists(final_model_path): # Use lexists to detect broken links too
                if os.path.islink(final_model_path):
                    print(f"Removing existing symlink: {final_model_path}")
                    os.unlink(final_model_path)
                elif os.path.isdir(final_model_path):
                     print(f"Removing existing directory: {final_model_path}")
                     shutil.rmtree(final_model_path)
                else:
                     print(f"Removing existing file: {final_model_path}")
                     os.remove(final_model_path)

            # Create the new symlink (only on non-Windows)
            if os.name != 'nt':
                 # Verify source directory exists before creating link
                 if os.path.isdir(expected_checkpoint_path):
                      os.symlink(expected_checkpoint_path, final_model_path, target_is_directory=True)
                      print(f"Successfully created 'final_model' symlink.")
                 else:
                      print(f"!!! Source checkpoint directory NOT FOUND: {expected_checkpoint_path}. Cannot create symlink.")
            else: # Basic copy for Windows
                 if os.path.isdir(expected_checkpoint_path):
                      shutil.copytree(expected_checkpoint_path, final_model_path)
                      print(f"Successfully created 'final_model' copy (Windows).")
                 else:
                      print(f"!!! Source checkpoint directory NOT FOUND: {expected_checkpoint_path}. Cannot create copy.")
            # --- End Simplified Symlink Creation ---

        except Exception as e:
            print(f"!!! ERROR during final checkpoint saving or symlink creation: {e}")
            import traceback
            traceback.print_exc()

    if summary_writer:
        summary_writer.close()

    return make_inference_fn, params, training_metrics 