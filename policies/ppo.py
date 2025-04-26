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
    brax_log_dir: str = "/tmp/mjx_brax_checkpoints",
    restore_checkpoint_path: Optional[str] = None,
    use_domain_randomization: bool = False,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    summary_writer: Optional[tensorboard.SummaryWriter] = None,
) -> Tuple[Callable, Any, Dict[str, Any]]:
    """
    Train a policy using PPO, relying on Brax internal checkpointing.

    Args:
        env: Environment to train on
        eval_env: Environment to evaluate on (if None, uses env)
        config: Configuration for training (if None, uses default)
        seed: Random seed
        brax_log_dir: Base directory where Brax will save checkpoints (inside a subdir) and potentially other logs.
        restore_checkpoint_path: Specific Brax checkpoint directory (e.g., .../checkpoints/000...) to restore from.
        use_domain_randomization: Whether to use domain randomization
        progress_fn: Function to call with training progress
        summary_writer: TensorBoard SummaryWriter for logging

    Returns:
        Tuple of (make_inference_fn, final_params, training_metrics)
    """
    # Load default config if none provided
    if config is None:
        from configs.default_configs import get_ppo_config
        config = get_ppo_config()

    # Ensure Brax log directory exists
    os.makedirs(brax_log_dir, exist_ok=True)
    print(f"Brax checkpoint log directory: {brax_log_dir}")

    # Keep default progress function for printing/TensorBoard, but remove saving logic
    if progress_fn is None:
        times = [datetime.now()] # Keep timing info
        print("Using default progress function for console/TensorBoard logging.")
        def default_progress(step, metrics):
            # Log timing only after first step completes (JIT compilation)
            if step > 0 and len(times) == 1:
                 times.append(datetime.now())
                 print(f"First step time (incl. JIT): {times[-1] - times[0]}")
            elif step > 0:
                 times.append(datetime.now())


            if summary_writer:
                for key, value in metrics.items():
                    try:
                        # Ensure value is scalar and finite for TensorBoard
                        if jp.isscalar(value) and jp.isfinite(value):
                             summary_writer.scalar(key, value, step)
                    except Exception as e:
                        print(f"Warning: Could not log metric '{key}' to TensorBoard: {e}")

            # Print key metrics to console
            current_eval_reward = metrics.get('eval/episode_reward', jp.nan)
            current_eval_reward_std = metrics.get('eval/episode_reward_std', 0)
            print(f"\n--- Step: {step}/{config.num_timesteps} ---")
            print(f"Eval Reward: {current_eval_reward:.3f} +/- {current_eval_reward_std:.3f}")
            if len(times) > 1:
                 print(f"Total training time: {(times[-1] - times[1]).total_seconds()/60:.2f} minutes")
            # Optionally print losses if needed:
            # actor_loss = metrics.get('loss/actor', metrics.get('ppo/actor_loss', jp.nan))
            # critic_loss = metrics.get('loss/critic', metrics.get('ppo/value_loss', jp.nan))
            # print(f"Losses (Actor/Critic): {actor_loss:.4f} / {critic_loss:.4f}")


        progress_fn = default_progress

    # Configure network factory
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.network.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=config.network.value_hidden_layer_sizes
    )

    # Configure training function
    from utils.domain_rand import domain_randomize

    # Create the partial function WITHOUT restore_checkpoint_path initially
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

    # Prepare the final keyword arguments for the train call
    train_kwargs = {
        "environment": env,
        "progress_fn": progress_fn,
        "eval_env": eval_env,
    }

    # Conditionally add the restore path to the arguments passed to ppo.train
    if restore_checkpoint_path is not None:
        print(f"Passing restore_checkpoint_path to ppo.train: {restore_checkpoint_path}")
        train_kwargs["restore_checkpoint_path"] = restore_checkpoint_path
    else:
        print("Starting fresh training (no restore_checkpoint_path provided)")

    # Train the policy using the potentially modified kwargs
    # Note: We are calling the ORIGINAL ppo.train via the partial function,
    # but only adding restore_checkpoint_path to the call if needed.
    # The 'initial_params' logic is removed.
    make_inference_fn, params, training_metrics = train_fn_partial(**train_kwargs)

    # Save the checkpoint after training
    if brax_log_dir:
        step = config.num_timesteps

        # Save network config and parameters - PASS BASE DIR to save()
        network_config = ppo_checkpoint.network_config(
            observation_size=env.observation_size,
            action_size=env.action_size,
            normalize_observations=config.normalize_observations,
            network_factory=make_networks_factory
        )
        try:
            # Let save handle creating the step directory inside brax_log_dir
            # Capture the actual path where the checkpoint was saved
            saved_checkpoint_path = ppo_checkpoint.save(
                path=brax_log_dir, # Pass the base directory
                step=step,
                params=params,
                config=network_config
            )
            print(f"Saved checkpoint to {saved_checkpoint_path}") # Use the returned path

            # Create a "final_model" symlink to the actual saved checkpoint directory
            final_model_path = os.path.join(brax_log_dir, "final_model")
            # Remove existing symlink/dir first (important!)
            if os.path.exists(final_model_path):
                if os.path.islink(final_model_path):
                    os.unlink(final_model_path)
                else: # If it's somehow a directory, remove it
                    import shutil
                    shutil.rmtree(final_model_path)

            # Create symlink on Unix or directory copy on Windows
            # Point to the actual saved path returned by save()
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copytree(saved_checkpoint_path, final_model_path)
                print(f"Created final_model copy at {final_model_path}")
            else:  # Unix-like
                # Use the correct target path returned by save()
                os.symlink(saved_checkpoint_path, final_model_path, target_is_directory=True)
                print(f"Created final_model symlink pointing to {saved_checkpoint_path}")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint or create final_model reference: {e}")
            # Add traceback for more details if needed
            # import traceback
            # traceback.print_exc()


    if summary_writer:
        summary_writer.close()

    return make_inference_fn, params, training_metrics 