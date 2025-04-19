"""
Default configurations for environments and training.
"""

from ml_collections import config_dict


def get_quadruped_config():
    """
    Get default configuration for quadruped environment.
    
    Returns:
        Configuration dictionary
    """
    config = config_dict.ConfigDict()
    config.rewards = config_dict.ConfigDict()
    config.rewards.tracking_sigma = 0.25
    
    # Default reward scales
    config.rewards.scales = config_dict.ConfigDict()
    config.rewards.scales.tracking_lin_vel = 1.0
    config.rewards.scales.tracking_ang_vel = 0.5
    config.rewards.scales.lin_vel_z = -2.0
    config.rewards.scales.ang_vel_xy = -0.05
    config.rewards.scales.orientation = -0.05
    config.rewards.scales.torques = -0.0001
    config.rewards.scales.action_rate = -0.01
    config.rewards.scales.stand_still = -0.1
    config.rewards.scales.feet_air_time = 2.0
    config.rewards.scales.foot_slip = -0.05
    config.rewards.scales.termination = -1.0
    
    return config


def get_ppo_config():
    """
    Get default configuration for PPO training.
    
    Returns:
        Configuration dictionary
    """
    config = config_dict.ConfigDict()
    
    # PPO hyperparameters
    config.num_timesteps = 100_000_000
    config.num_envs = 2048
    config.num_evals = 100 # How often to evaluate (triggers progress_fn and policy_params_fn)
    # Save a 'step_X' checkpoint approx every N steps. Set high to save infrequently.
    # Set to -1 or 0 to disable periodic step checkpoints altogether.
    config.periodic_checkpoint_save_interval_steps = 5_000_000 # e.g., save step_X roughly 4 times for 20M steps
    config.reward_scaling = 0.1
    config.episode_length = 1000
    config.normalize_observations = True
    config.action_repeat = 1
    config.unroll_length = 10
    config.num_minibatches = 32
    config.num_updates_per_batch = 4
    config.discounting = 0.97
    config.learning_rate = 3e-4
    config.entropy_cost = 1e-2
    config.batch_size = 256
    
    # Network architecture
    config.network = config_dict.ConfigDict()
    config.network.policy_hidden_layer_sizes = (128, 128, 128, 128)
    config.network.value_hidden_layer_sizes = (128, 128, 128, 128)
    
    return config 