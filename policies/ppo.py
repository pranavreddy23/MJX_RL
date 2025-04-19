"""
PPO policy implementation for MJX environments.
"""

import functools
import os
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jp
from flax.metrics import tensorboard
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from datetime import datetime
import pickle
import pandas as pd

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from brax.envs.base import Env
from ml_collections import ConfigDict


def train_ppo(
    env: Env,
    eval_env: Optional[Env] = None,
    config: Optional[ConfigDict] = None,
    seed: int = 0,
    checkpoint_dir: str = "/tmp/mjx_checkpoints",
    restore_checkpoint_path: Optional[str] = None,
    use_domain_randomization: bool = False,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    summary_writer: Optional[tensorboard.SummaryWriter] = None,
) -> Tuple[Callable, Any, Dict[str, Any]]:
    """
    Train a policy using PPO.
    
    Args:
        env: Environment to train on
        eval_env: Environment to evaluate on (if None, uses env)
        config: Configuration for training (if None, uses default)
        seed: Random seed
        checkpoint_dir: Directory to save checkpoints
        restore_checkpoint_path: Path to checkpoint to restore from
        use_domain_randomization: Whether to use domain randomization
        progress_fn: Function to call with training progress
        summary_writer: TensorBoard SummaryWriter for logging
        
    Returns:
        Tuple of (make_inference_fn, params, training_metrics)
    """
    # Load default config if none provided
    if config is None:
        from configs.default_configs import get_ppo_config
        config = get_ppo_config()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create metrics directory if it doesn't exist (for CSV)
    metrics_dir = os.path.join(checkpoint_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # --- Variables for tracking best model ---
    best_eval_reward = -jp.inf
    best_step = -1
    # Use a simple checkpointer for best model saving, overwriting the previous best
    best_checkpointer = ocp.PyTreeCheckpointer()
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint')
    
    # --- Variable for periodic saving frequency ---
    last_periodic_save_step = -1
    
    # Create default progress function if none provided
    if progress_fn is None:
        # Initialize tracking variables for CSV
        steps = []
        times = [datetime.now()]
        metrics_data = {'step': [], 'time_minutes': [], 'eval/episode_reward': [], 'eval/episode_reward_std': []}
        # Add common loss keys, will only populate if found
        loss_keys_to_track = [
            'train/actor_loss', 'loss/actor', 'ppo/actor_loss',
            'train/critic_loss', 'loss/critic', 'ppo/critic_loss', 'ppo/value_loss',
            'train/entropy_loss', 'loss/entropy', 'ppo/entropy_loss', 'ppo/entropy'
        ]
        for k in loss_keys_to_track:
             metrics_data[k] = []

        def default_progress(step, metrics):
            nonlocal best_eval_reward, best_step # Allow modification

            # Record time for first step to measure compilation time
            if step == 0:
                times.append(datetime.now())
                print(f"JIT compilation time: {times[1] - times[0]}")
                return
            
            # Record time
            times.append(datetime.now())
            current_time_minutes = (times[-1] - times[1]).total_seconds() / 60

            # Store basic info
            steps.append(step)
            metrics_data['step'].append(step)
            metrics_data['time_minutes'].append(current_time_minutes)

            # --- Log ALL metrics to TensorBoard ---
            if summary_writer:
                for key, value in metrics.items():
                    try:
                        # Ensure value is scalar and finite for TensorBoard
                        if jp.isscalar(value) and jp.isfinite(value):
                             summary_writer.scalar(key, value, step)
                    except Exception as e:
                        print(f"Warning: Could not log metric '{key}' to TensorBoard: {e}")

            # --- Extract and print key metrics ---
            current_eval_reward = metrics.get('eval/episode_reward', jp.nan)
            current_eval_reward_std = metrics.get('eval/episode_reward_std', 0)

            metrics_data['eval/episode_reward'].append(float(current_eval_reward))
            metrics_data['eval/episode_reward_std'].append(float(current_eval_reward_std))

            print(f"\n--- Step: {step}/{config.num_timesteps} ---")
            print(f"Eval Reward: {current_eval_reward:.3f} +/- {current_eval_reward_std:.3f}")

            # Log and store specific losses if found
            found_losses = False
            for key in loss_keys_to_track:
                 if key in metrics:
                    loss_val = metrics[key]
                    metrics_data[key].append(float(loss_val))
                    print(f"{key}: {loss_val:.4f}")
                    found_losses = True
                 else:
                     # Append NaN if key not present in this step's metrics
                     # Ensure list length matches steps list length
                     if len(metrics_data[key]) < len(steps):
                         metrics_data[key].append(jp.nan)

            print(f"Time since last step: {times[-1] - times[-2]}")
            print(f"Total training time: {times[-1] - times[1]}")

            # --- Check for new best reward ---
            if jp.isfinite(current_eval_reward) and current_eval_reward > best_eval_reward:
                print(f"*** New best eval reward: {current_eval_reward:.4f} at step {step} ***")
                best_eval_reward = current_eval_reward
                best_step = step
                # The saving happens in policy_params_fn when current_step == best_step

            # --- Save metrics to CSV ---
            try:
                # Create DataFrame ensuring all lists have the same length
                max_len = len(steps)
                data_for_df = {}
                for k, v_list in metrics_data.items():
                    # Pad with NaN if necessary (e.g., if losses appear later)
                    padded_list = v_list + [jp.nan] * (max_len - len(v_list))
                    data_for_df[k] = padded_list

                df = pd.DataFrame(data_for_df)
                df.to_csv(os.path.join(metrics_dir, 'training_metrics.csv'), index=False)
            except Exception as e:
                print(f"Warning: Failed to save metrics to CSV: {e}")

        progress_fn = default_progress
    
    # Function to save checkpoints during training
    def policy_params_fn(current_step, make_policy, params):
        nonlocal last_periodic_save_step # Access the step tracker

        # --- Periodic saving based on step interval ---
        save_interval = config.get('periodic_checkpoint_save_interval_steps', -1)
        should_save_periodic = (
            save_interval > 0 and
            current_step >= save_interval and # Avoid saving at step 0 unnecessarily
            (current_step - last_periodic_save_step >= save_interval)
        )

        if should_save_periodic:
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(params)
            periodic_path = os.path.join(checkpoint_dir, f'step_{current_step}')
            try:
                orbax_checkpointer.save(periodic_path, params, force=True, save_args=save_args)
                print(f"Saved periodic checkpoint (interval): {periodic_path}")
                last_periodic_save_step = current_step # Update the tracker
            except Exception as e:
                print(f"Warning: Failed to save periodic checkpoint at step {current_step}: {e}")

        # --- Best checkpoint saving (still happens whenever a new best is found) ---
        # Save if this is the best step identified by progress_fn
        # Use simple checkpointer, forcing overwrite
        if current_step == best_step:
             try:
                save_args = orbax_utils.save_args_from_target(params)
                best_checkpointer.save(best_checkpoint_path, params, force=True, save_args=save_args)
                print(f"*** Saved new best checkpoint (Eval Reward: {best_eval_reward:.4f}) at step {current_step} to {best_checkpoint_path} ***")
             except Exception as e:
                 print(f"Warning: Failed to save best checkpoint at step {current_step}: {e}")

    # Configure network factory
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.network.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=config.network.value_hidden_layer_sizes
    )
    
    # Configure training function
    from utils.domain_rand import domain_randomize
    
    train_fn = functools.partial(
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
        policy_params_fn=policy_params_fn,
        seed=seed,
    )
    
    # Add domain randomization if requested
    if use_domain_randomization:
        train_fn = functools.partial(train_fn, randomization_fn=domain_randomize)
    
    # Add checkpoint restoration if requested
    if restore_checkpoint_path:
        print(f"Restoring training from checkpoint: {restore_checkpoint_path}")
        train_fn = functools.partial(train_fn, restore_checkpoint_path=restore_checkpoint_path)
    
    # Train the policy
    make_inference_fn, params, training_metrics = train_fn(
        environment=env,
        progress_fn=progress_fn,
        eval_env=eval_env,
    )
    
    # Save final model parameters regardless of performance
    # This is useful if training is stopped manually or finishes
    final_model_path = os.path.join(checkpoint_dir, 'final_model')
    try:
        final_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        final_checkpointer.save(final_model_path, params, force=True, save_args=save_args)
        print(f"Saved final model parameters to {final_model_path}")
    except Exception as e:
        print(f"Warning: Failed to save final model: {e}")

    # Close the summary writer if it was provided
    if summary_writer:
        summary_writer.close()

    return make_inference_fn, params, training_metrics 