"""
PPO policy implementation for MJX environments.
"""

import functools
import os
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from datetime import datetime
import pickle

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from brax.envs.base import Env
from ml_collections import ConfigDict


def make_inference_fn():
    """
    Creates a function that builds an inference function from parameters.
    
    Returns:
        A function that takes parameters and returns an inference function
    """
    def inference_fn_factory(params):
        # Extract the policy network parameters
        # In Brax PPO, the parameters structure is typically:
        # (running_statistics, policy_params, value_params)
        running_statistics = params[0]
        policy_params = params[1]
        
        def inference_fn(obs, rng):
            # Normalize observations using running statistics
            normalized_obs = running_statistics.normalize(obs)
            
            # Apply the policy network
            # The policy network typically has a 'policy_network' attribute
            action_mean = policy_params.apply_fn(
                {'params': policy_params.params},
                normalized_obs
            )
            
            # For deterministic actions, just return the mean
            return action_mean, {'log_prob': jp.zeros(action_mean.shape[0])}
        
        return inference_fn
    
    return inference_fn_factory


def train_ppo(
    env: Env,
    eval_env: Optional[Env] = None,
    config: Optional[ConfigDict] = None,
    seed: int = 0,
    checkpoint_dir: str = "/tmp/mjx_checkpoints",
    restore_checkpoint_path: Optional[str] = None,
    use_domain_randomization: bool = False,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
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
        
    Returns:
        Tuple of (make_inference_fn, params, training_metrics)
    """
    # Load default config if none provided
    if config is None:
        from configs.default_configs import get_ppo_config
        config = get_ppo_config()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create metrics directory
    metrics_dir = os.path.join(checkpoint_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create default progress function if none provided
    if progress_fn is None:
        # Initialize tracking variables
        steps = []
        rewards = []
        reward_stds = []
        actor_losses = []
        critic_losses = []
        entropies = []
        times = [datetime.now()]
        
        def default_progress(step, metrics):
            # Record time for first step to measure compilation time
            if step == 0:
                times.append(datetime.now())
                print(f"JIT compilation time: {times[1] - times[0]}")
                return
            
            # Record time
            times.append(datetime.now())
            steps.append(step)
            
            # Print all available metrics keys for debugging
            print("\nAvailable metrics:", list(metrics.keys()))
            
            # Extract metrics
            if 'eval/episode_reward' in metrics:
                rewards.append(metrics['eval/episode_reward'])
                reward_stds.append(metrics.get('eval/episode_reward_std', 0))
            
            # Look for actor/critic losses with different possible keys
            actor_loss_keys = ['train/actor_loss', 'loss/actor', 'ppo/actor_loss']
            critic_loss_keys = ['train/critic_loss', 'loss/critic', 'ppo/critic_loss', 'ppo/value_loss']
            entropy_keys = ['train/entropy_loss', 'loss/entropy', 'ppo/entropy_loss', 'ppo/entropy']
            
            # Try to find actor loss
            for key in actor_loss_keys:
                if key in metrics:
                    actor_losses.append(metrics[key])
                    print(f"Actor Loss: {metrics[key]:.4f}")
                    break
            
            # Try to find critic loss
            for key in critic_loss_keys:
                if key in metrics:
                    critic_losses.append(metrics[key])
                    print(f"Critic Loss: {metrics[key]:.4f}")
                    break
            
            # Try to find entropy
            for key in entropy_keys:
                if key in metrics:
                    entropies.append(metrics[key])
                    print(f"Entropy: {metrics[key]:.4f}")
                    break
            
            # Print progress
            print(f"\n--- Step: {step}/{config.num_timesteps} ---")
            print(f"Reward: {metrics.get('eval/episode_reward', 'N/A'):.2f} ± "
                  f"{metrics.get('eval/episode_reward_std', 0):.2f}")
            print(f"Time since last step: {times[-1] - times[-2]}")
            print(f"Total training time: {times[-1] - times[1]}")
            
            # Plot progress if requested
            if config.show_progress_plot and len(steps) > 1:
                # Create figure with multiple subplots
                plt.figure(figsize=(15, 10))
                
                # Plot 1: Reward over steps
                plt.subplot(2, 2, 1)
                plt.errorbar(steps, rewards, yerr=reward_stds, capsize=2)
                plt.xlabel('Training steps')
                plt.ylabel('Reward')
                plt.title('Training Progress')
                
                # Plot 2: Reward over time
                plt.subplot(2, 2, 2)
                times_min = [(t - times[1]).total_seconds() / 60 for t in times[2:]]
                plt.plot(times_min, rewards)
                plt.xlabel('Training time (minutes)')
                plt.ylabel('Reward')
                plt.title('Reward vs. Time')
                
                # Plot 3: Actor-Critic losses
                plt.subplot(2, 2, 3)
                if actor_losses:
                    plt.plot(steps, actor_losses, label='Actor Loss')
                if critic_losses:
                    plt.plot(steps, critic_losses, label='Critic Loss')
                if actor_losses or critic_losses:  # Only add legend if we have data
                    plt.legend()
                plt.xlabel('Training steps')
                plt.ylabel('Loss')
                plt.title('Actor-Critic Losses')
                
                # Plot 4: Entropy
                plt.subplot(2, 2, 4)
                if entropies:
                    plt.plot(steps, entropies, label='Entropy')
                    plt.legend()
                plt.xlabel('Training steps')
                plt.ylabel('Value')
                plt.title('Policy Entropy')
                
                plt.tight_layout()
                plt.savefig(os.path.join(metrics_dir, 'training_progress.png'))
                plt.close()
                
                # Save metrics to CSV for later analysis
                try:
                    import pandas as pd
                    data = {
                        'step': steps,
                        'reward': rewards,
                        'reward_std': reward_stds,
                    }
                    
                    if actor_losses:
                        data['actor_loss'] = actor_losses
                    if critic_losses:
                        data['critic_loss'] = critic_losses
                    if entropies:
                        data['entropy'] = entropies
                        
                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(metrics_dir, 'training_metrics.csv'), index=False)
                except ImportError:
                    print("pandas not installed, skipping CSV export")
        
        progress_fn = default_progress
    
    # Function to save checkpoints during training
    def policy_params_fn(current_step, make_policy, params):
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = os.path.join(checkpoint_dir, f'{current_step}')
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
    
    # Configure network factory
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.network.policy_hidden_layer_sizes
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
        train_fn = functools.partial(train_fn, restore_checkpoint_path=restore_checkpoint_path)
    
    # Train the policy
    make_inference_fn, params, training_metrics = train_fn(
        environment=env,
        progress_fn=progress_fn,
        eval_env=eval_env,
    )
    
    # Save final model
    model_path = os.path.join(checkpoint_dir, 'final_model')
    model.save_params(model_path, params)
    
    # Also save the make_inference_fn for later use
    inference_fn_path = os.path.join(checkpoint_dir, 'make_inference_fn.pkl')
    with open(inference_fn_path, 'wb') as f:
        pickle.dump(make_inference_fn, f)
    
    return make_inference_fn, params, training_metrics 