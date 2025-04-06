"""
Utilities for saving and loading checkpoints.
"""

import os
from typing import Any, Dict, List, Optional

import jax
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from brax.io import model


def save_params(checkpoint_dir: str, params: Any, step: Optional[int] = None):
    """
    Save parameters to a checkpoint.
    
    Args:
        checkpoint_dir: Directory to save the checkpoint
        params: Parameters to save
        step: Training step (if None, saves as 'final')
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    
    if step is not None:
        path = os.path.join(checkpoint_dir, f'{step}')
    else:
        path = os.path.join(checkpoint_dir, 'final')
    
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)
    
    # Also save in Brax model format for compatibility
    model_path = os.path.join(checkpoint_dir, 'model')
    model.save_params(model_path, params)


def load_params(checkpoint_path: str) -> Any:
    """
    Load parameters from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        
    Returns:
        Loaded parameters
    """
    # Try loading as a directory first (Orbax format)
    if os.path.isdir(checkpoint_path):
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        return orbax_checkpointer.restore(checkpoint_path)
    
    # If not a directory, try loading as a Brax model
    return model.load_params(checkpoint_path)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Check for numeric checkpoints (training steps)
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        try:
            step = int(item)
            checkpoints.append((step, os.path.join(checkpoint_dir, item)))
        except ValueError:
            continue
    
    if checkpoints:
        # Sort by step number and return the latest
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]
    
    # Check for 'final' checkpoint
    final_path = os.path.join(checkpoint_dir, 'final')
    if os.path.exists(final_path):
        return final_path
    
    # Check for Brax model
    model_path = os.path.join(checkpoint_dir, 'model')
    if os.path.exists(model_path):
        return model_path
    
    return None 