"""
Monitoring utilities for training visualization and metrics tracking.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime

from brax.envs.base import Env, State
from brax.training.types import Params

from utils.rendering import render_policy


class TrainingMonitor:
    """Monitor training progress with detailed metrics and visualizations."""
    
    def __init__(
        self,
        env: Env,
        checkpoint_dir: str,
        render_interval: int = 1000000,  # Render every N steps
        save_interval: int = 1000000,    # Save metrics every N steps
        max_render_episodes: int = 3,
        render_steps: int = 300,
    ):
        self.env = env
        self.checkpoint_dir = checkpoint_dir
        self.render_interval = render_interval
        self.save_interval = save_interval
        self.max_render_episodes = max_render_episodes
        self.render_steps = render_steps
        
        # Create directories
        self.metrics_dir = os.path.join(checkpoint_dir, 'metrics')
        self.renders_dir = os.path.join(checkpoint_dir, 'renders')
        self.plots_dir = os.path.join(checkpoint_dir, 'plots')
        
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.renders_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.steps = []
        self.times = [datetime.now()]
        self.rewards = []
        self.reward_stds = []
        
        # Actor-critic metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.kls = []
        
        # Environment metrics
        self.env_metrics = {}
        
        # Compilation flag
        self.compilation_done = False
    
    def progress_callback(
        self, 
        step: int, 
        metrics: Dict[str, Any], 
        make_inference_fn=None, 
        params=None
    ):
        """Callback function to track training progress."""
        # Record time for first step to measure compilation time
        if step == 0:
            self.times.append(datetime.now())
            print(f"JIT compilation time: {self.times[1] - self.times[0]}")
            return
        
        # Record time and step
        current_time = datetime.now()
        self.times.append(current_time)
        self.steps.append(step)
        
        # Extract and store metrics
        if 'eval/episode_reward' in metrics:
            self.rewards.append(metrics['eval/episode_reward'])
            self.reward_stds.append(metrics.get('eval/episode_reward_std', 0))
        
        # Extract actor-critic metrics
        if 'train/actor_loss' in metrics:
            self.actor_losses.append(metrics['train/actor_loss'])
        if 'train/critic_loss' in metrics:
            self.critic_losses.append(metrics['train/critic_loss'])
        if 'train/entropy_loss' in metrics:
            self.entropies.append(metrics['train/entropy_loss'])
        if 'train/kl' in metrics:
            self.kls.append(metrics['train/kl'])
        
        # Extract environment-specific metrics
        for key, value in metrics.items():
            if key.startswith('eval/'):
                if key not in self.env_metrics:
                    self.env_metrics[key] = []
                self.env_metrics[key].append(value)
        
        # Print progress
        print(f"\n--- Step: {step} ---")
        print(f"Reward: {metrics.get('eval/episode_reward', 'N/A'):.2f} ± "
              f"{metrics.get('eval/episode_reward_std', 0):.2f}")
        print(f"Actor Loss: {metrics.get('train/actor_loss', 'N/A')}")
        print(f"Critic Loss: {metrics.get('train/critic_loss', 'N/A')}")
        print(f"Time since last step: {current_time - self.times[-2]}")
        print(f"Total training time: {current_time - self.times[1]}")
        
        # Create and save plots
        if len(self.steps) > 1:
            self._create_plots(step)
        
        # Render policy if interval is reached and inference function is available
        if make_inference_fn is not None and params is not None:
            if step % self.render_interval == 0 or step == 1:
                self._render_policy(step, make_inference_fn, params)
        
        # Save metrics if interval is reached
        if step % self.save_interval == 0:
            self._save_metrics(step)
    
    def _create_plots(self, step: int):
        """Create and save training progress plots."""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Reward over steps
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.errorbar(self.steps, self.rewards, yerr=self.reward_stds, capsize=2)
        ax1.set_xlabel('Training steps')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress')
        
        # Plot 2: Reward over time
        ax2 = fig.add_subplot(2, 2, 2)
        times_min = [(t - self.times[1]).total_seconds() / 60 for t in self.times[2:]]
        ax2.plot(times_min, self.rewards)
        ax2.set_xlabel('Training time (minutes)')
        ax2.set_ylabel('Reward')
        ax2.set_title('Reward vs. Time')
        
        # Plot 3: Actor-Critic losses
        ax3 = fig.add_subplot(2, 2, 3)
        if self.actor_losses:
            ax3.plot(self.steps, self.actor_losses, label='Actor Loss')
        if self.critic_losses:
            ax3.plot(self.steps, self.critic_losses, label='Critic Loss')
        ax3.set_xlabel('Training steps')
        ax3.set_ylabel('Loss')
        ax3.set_title('Actor-Critic Losses')
        ax3.legend()
        
        # Plot 4: Entropy and KL divergence
        ax4 = fig.add_subplot(2, 2, 4)
        if self.entropies:
            ax4.plot(self.steps, self.entropies, label='Entropy')
        if self.kls:
            ax4.plot(self.steps, self.kls, label='KL Divergence')
        ax4.set_xlabel('Training steps')
        ax4.set_ylabel('Value')
        ax4.set_title('Policy Metrics')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'training_progress_{step}.png'))
        plt.savefig(os.path.join(self.plots_dir, 'latest_progress.png'))
        plt.close(fig)
        
        # Create additional plots for environment-specific metrics
        if self.env_metrics:
            fig = plt.figure(figsize=(15, 10))
            num_metrics = len(self.env_metrics)
            rows = (num_metrics + 1) // 2
            
            for i, (key, values) in enumerate(self.env_metrics.items()):
                ax = fig.add_subplot(rows, 2, i+1)
                ax.plot(self.steps, values)
                ax.set_xlabel('Training steps')
                ax.set_ylabel(key.split('/')[-1])
                ax.set_title(key)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'env_metrics_{step}.png'))
            plt.savefig(os.path.join(self.plots_dir, 'latest_env_metrics.png'))
            plt.close(fig)
    
    def _render_policy(self, step: int, make_inference_fn, params):
        """Render the current policy and save frames."""
        try:
            import mediapy as media
            
            inference_fn = make_inference_fn(params)
            jit_inference_fn = jax.jit(inference_fn)
            
            # JIT the environment functions
            jit_reset = jax.jit(self.env.reset)
            jit_step = jax.jit(self.env.step)
            
            # Render multiple episodes with different seeds
            for episode in range(self.max_render_episodes):
                # Initialize the environment
                rng = jax.random.PRNGKey(episode)
                state = jit_reset(rng)
                
                # For quadruped, set different commands for visualization
                if hasattr(state, 'info') and 'command' in state.info:
                    if episode == 0:
                        # Forward movement
                        state.info['command'] = jp.array([1.0, 0.0, 0.0])
                    elif episode == 1:
                        # Sideways movement
                        state.info['command'] = jp.array([0.0, 1.0, 0.0])
                    elif episode == 2:
                        # Turning
                        state.info['command'] = jp.array([0.5, 0.0, 0.5])
                
                # Collect trajectory
                rollout = [state.pipeline_state]
                
                for i in range(self.render_steps):
                    act_rng, rng = jax.random.split(rng)
                    action, _ = jit_inference_fn(state.obs, act_rng)
                    state = jit_step(state, action)
                    rollout.append(state.pipeline_state)
                
                # Render and save video
                render_every = 2
                frames = self.env.render(rollout[::render_every], camera='track')
                
                # Save video
                video_path = os.path.join(
                    self.renders_dir, 
                    f'policy_step_{step}_episode_{episode}.mp4'
                )
                media.write_video(video_path, frames, fps=1.0 / self.env.dt / render_every)
                
                # Also save the latest video with a consistent name for easy access
                latest_path = os.path.join(
                    self.renders_dir, 
                    f'latest_episode_{episode}.mp4'
                )
                media.write_video(latest_path, frames, fps=1.0 / self.env.dt / render_every)
                
                print(f"Saved video for step {step}, episode {episode}")
        
        except Exception as e:
            print(f"Error rendering policy: {e}")
    
    def _save_metrics(self, step: int):
        """Save metrics to disk."""
        import json
        
        metrics = {
            'steps': self.steps,
            'rewards': self.rewards,
            'reward_stds': self.reward_stds,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropies': self.entropies,
            'kls': self.kls,
            'env_metrics': self.env_metrics,
            'times': [t.isoformat() for t in self.times[1:]],
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    metrics[key][k] = [float(x) for x in v]
            elif isinstance(value, list) and value and hasattr(value[0], 'item'):
                metrics[key] = [float(x) for x in value]
        
        with open(os.path.join(self.metrics_dir, f'metrics_{step}.json'), 'w') as f:
            json.dump(metrics, f)
        
        # Also save the latest metrics with a consistent name
        with open(os.path.join(self.metrics_dir, 'latest_metrics.json'), 'w') as f:
            json.dump(metrics, f) 