"""
Environment registration and factory functions.
"""

from environments.quadruped import BarkourEnv

# Environment registry
_ENVIRONMENTS = {
    "quadruped": BarkourEnv,
}

def get_environment(env_name, **kwargs):
    """
    Get an environment by name.
    
    Args:
        env_name: Name of the environment
        **kwargs: Additional arguments to pass to the environment constructor
        
    Returns:
        An environment instance
    """
    if env_name not in _ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return _ENVIRONMENTS[env_name](**kwargs)

def register_environment(name, env_class):
    """
    Register a new environment.
    
    Args:
        name: Name of the environment
        env_class: Environment class
    """
    _ENVIRONMENTS[name] = env_class 