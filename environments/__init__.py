"""
Environment registration and factory functions.
"""

from environments.quadruped import BarkourEnv
from environments.humanoid import Humanoid  # Import the new Humanoid environment

# Environment registry
_ENVIRONMENTS = {
    "quadruped": BarkourEnv,
}

def get_environment(env_name: str, **kwargs):
    """Get an environment by name.
    
    Args:
        env_name: Name of the environment to get.
        **kwargs: Additional keyword arguments to pass to the environment.
        
    Returns:
        An environment instance.
        
    Raises:
        ValueError: If the requested environment is not recognized.
    """
    if env_name.lower() == "quadruped":
        return BarkourEnv(**kwargs)
    elif env_name.lower() == "humanoid":
        return Humanoid(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def register_environment(name, env_class):
    """
    Register a new environment.
    
    Args:
        name: Name of the environment
        env_class: Environment class
    """
    _ENVIRONMENTS[name] = env_class 