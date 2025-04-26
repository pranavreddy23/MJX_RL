#!/usr/bin/env python3
"""
Interactively visualize and control a trained policy using WASD keys.
Loads checkpoints saved by Brax internal mechanism using load_policy.

Controls:
  W/S: Increase/Decrease forward velocity (vx)
  A/D: Increase/Decrease sideways velocity (vy)
  Q/E: Increase/Decrease turning velocity (ang_vel)
  Space: Reset command velocities to zero
  R: Reset environment state
  ESC: Quit
"""

import os
import argparse
import jax
import jax.numpy as jp
import numpy as np
import time
import functools
from typing import Optional, List # Added for typing
import traceback # Keep for general error handling

# --- Viewer and MuJoCo/MJX ---
import mujoco_viewer
import mujoco
from mujoco import mjx

# --- Brax imports for loading ---
from brax.training.agents.ppo import networks as ppo_networks
# --- Use the dedicated loader ---
from brax.training.agents.ppo.checkpoint import load_policy
from brax.envs.base import State # Need State for typing

# --- Project specific ---
from environments import get_environment
from configs.default_configs import get_ppo_config # Still need config for network sizes

# --- Import or Copy Checkpoint Finder ---
try:
    # Assumes main is in the same parent directory or PYTHONPATH
    from main import find_brax_checkpoint
    print("Imported find_brax_checkpoint from main.py")
except ImportError:
    print("Warning: Could not import find_brax_checkpoint from main.py.")
    print("Using fallback checkpoint finding logic.")
    # --- Fallback Checkpoint Finding Logic (copy from main.py, ensure it's updated) ---
    def find_brax_checkpoint(brax_log_dir: str) -> Optional[str]:
        """
        Finds the latest valid Brax PPO checkpoint directory within the log directory.
        Checks for step-numbered directories containing '_METADATA' and 'ppo_network_config.json'.
        Prioritizes 'final_model' if it exists and points to a valid checkpoint.
        """
        latest_valid_step = -1
        latest_valid_ckpt_path = None
        print(f"Scanning for Brax checkpoints in: {brax_log_dir}")
        if not os.path.isdir(brax_log_dir): return None
        for item in os.listdir(brax_log_dir):
            path = os.path.join(brax_log_dir, item)
            if os.path.isdir(path) and item.isdigit():
                try:
                    step = int(item)
                    metadata_path = os.path.join(path, '_METADATA')
                    config_path = os.path.join(path, 'ppo_network_config.json') # Correct config name
                    if os.path.exists(metadata_path) and os.path.exists(config_path):
                        if step > latest_valid_step:
                            latest_valid_step = step
                            latest_valid_ckpt_path = path
                            print(f"  Found valid step checkpoint: {path} (Step: {step})")
                except ValueError: continue
        final_model_path = os.path.join(brax_log_dir, 'final_model')
        if os.path.exists(final_model_path):
             print(f"Found potential 'final_model' at {final_model_path}")
             real_final_model_path = os.path.realpath(final_model_path)
             if os.path.isdir(real_final_model_path):
                  metadata_path = os.path.join(real_final_model_path, '_METADATA')
                  config_path = os.path.join(real_final_model_path, 'ppo_network_config.json')
                  if os.path.exists(metadata_path) and os.path.exists(config_path):
                       print(f"Using 'final_model' pointing to valid checkpoint directory: {real_final_model_path}")
                       return real_final_model_path
                  else: print(f"Ignoring 'final_model' link/dir {final_model_path} as target {real_final_model_path} is missing required files.")
             else: print(f"Ignoring 'final_model' at {final_model_path} as it's not a valid directory or link to one.")
        if latest_valid_ckpt_path: print(f"Using latest valid step checkpoint found: {latest_valid_ckpt_path} (Step: {latest_valid_step})")
        else: print(f"No valid step-numbered Brax checkpoints found in {brax_log_dir}.")
        return latest_valid_ckpt_path
    # --- End Fallback ---


# --- Global dictionary to track key states ---
key_states = {
    'W': False, 'S': False, 'A': False, 'D': False,
    'Q': False, 'E': False, 'SPACE': False, 'R': False,
    'ESC': False
}

def key_callback(keycode, action, mods):
    """Callback function for keyboard events."""
    global key_states
    key_map = {
        mujoco.glfw.KEY_W: 'W',
        mujoco.glfw.KEY_S: 'S',
        mujoco.glfw.KEY_A: 'A',
        mujoco.glfw.KEY_D: 'D',
        mujoco.glfw.KEY_Q: 'Q',
        mujoco.glfw.KEY_E: 'E',
        mujoco.glfw.KEY_SPACE: 'SPACE',
        mujoco.glfw.KEY_R: 'R',
        mujoco.glfw.KEY_ESCAPE: 'ESC',
    }
    key_char = key_map.get(keycode)

    if key_char:
        if action == mujoco.glfw.PRESS:
            key_states[key_char] = True
        elif action == mujoco.glfw.RELEASE:
            key_states[key_char] = False

def main():
    parser = argparse.ArgumentParser(description="Interactive policy visualization using Brax checkpoints")
    # Use the new directory argument
    parser.add_argument("--brax_checkpoint_dir", type=str, default="/tmp/mjx_brax_checkpoints", help="Base directory where Brax saved checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific Brax checkpoint directory (e.g., .../000...) or 'final_model'")
    parser.add_argument("--env", type=str, default="quadruped", help="Environment name")
    parser.add_argument("--scene", type=str, default=None, help="Scene file (for quadruped)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for environment reset")
    parser.add_argument("--max_vx", type=float, default=1.5, help="Max target forward velocity")
    parser.add_argument("--max_vy", type=float, default=1.0, help="Max target sideways velocity")
    parser.add_argument("--max_ang_vel", type=float, default=1.0, help="Max target angular velocity")
    parser.add_argument("--vel_increment", type=float, default=0.1, help="Velocity increment per key press")
    parser.add_argument("--ang_increment", type=float, default=0.1, help="Angular velocity increment per key press")
    parser.add_argument("--decay_factor", type=float, default=0.95, help="Factor to decay velocity when no keys pressed")

    args = parser.parse_args()

    # --- Determine checkpoint path (using find_brax_checkpoint) ---
    checkpoint_to_load = args.checkpoint
    if not checkpoint_to_load:
        checkpoint_to_load = find_brax_checkpoint(args.brax_checkpoint_dir)
        if checkpoint_to_load:
            print(f"Found Brax checkpoint to load: {checkpoint_to_load}")
        else:
            raise FileNotFoundError(f"No specific checkpoint provided and no valid Brax checkpoint found in {args.brax_checkpoint_dir}")
    elif checkpoint_to_load.lower() == 'final_model':
         # If user specified 'final_model', resolve it
         resolved_path = find_brax_checkpoint(args.brax_checkpoint_dir) # This function now handles final_model priority
         if resolved_path and os.path.basename(os.path.realpath(resolved_path)) == 'final_model':
              checkpoint_to_load = resolved_path
              print(f"Resolved 'final_model' to: {checkpoint_to_load}")
         elif resolved_path: # find_brax_checkpoint returned latest step instead
             print(f"Warning: 'final_model' not found or invalid. Using latest step: {resolved_path}")
             checkpoint_to_load = resolved_path
         else:
              raise FileNotFoundError(f"Specified 'final_model' but no valid checkpoint found in {args.brax_checkpoint_dir}")
    else:
        # User specified a direct path, ensure it's resolved (in case it's a symlink itself)
        checkpoint_to_load = os.path.realpath(args.checkpoint)
        print(f"Using user-specified checkpoint path (resolved): {checkpoint_to_load}")

    if not os.path.exists(checkpoint_to_load) or not os.path.isdir(checkpoint_to_load):
         raise FileNotFoundError(f"Checkpoint path not found or not a directory: {checkpoint_to_load}")

    print(f"JAX devices: {jax.devices()}")

    # --- Load Config and Create Environment ---
    config = get_ppo_config() # Needed for network factory structure
    env_kwargs = {}
    if args.env == "quadruped" and args.scene is not None:
        env_kwargs["scene_file"] = args.scene
    env = get_environment(args.env, **env_kwargs)
    print(f"Environment '{args.env}' loaded.")

    # --- Create Network Factory ---
    # load_policy needs this to reconstruct the network based on the saved config
    make_networks_factory_eval = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.network.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=config.network.value_hidden_layer_sizes
        # NOTE: DO NOT pass normalize_observations here, load_policy reads it from the saved config
    )

    # --- Load the policy using the dedicated function ---
    print(f"Loading policy using brax.training.agents.ppo.checkpoint.load_policy from: {checkpoint_to_load}")
    try:
        inference_fn = load_policy(
            path=checkpoint_to_load,
            network_factory=make_networks_factory_eval,
            deterministic=True # Typically True for visualization/evaluation
        )
        jit_inference_fn = jax.jit(inference_fn)
        print("Policy loaded and inference function ready.")
    except FileNotFoundError as e:
         print(f"ERROR loading policy: {e}")
         print("Ensure the checkpoint path is correct and contains '_METADATA' and 'ppo_network_config.json'.")
         raise
    except Exception as e:
        print(f"Error loading policy with Brax loader: {e}")
        traceback.print_exc()
        raise

    # --- JIT and Environment Setup ---
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # --- Initialize State ---
    rng = jax.random.PRNGKey(args.seed)
    state = jit_reset(rng)
    mjx_model = env.sys.mj_model
    mjx_data = state.pipeline_state

    # --- Setup Viewer ---
    try:
        viewer = mujoco_viewer.MujocoViewer(mjx_model, mjx_data, title="Interactive Policy Viewer", width=1200, height=900)
        viewer.key_callback = key_callback
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
    except Exception as e:
        print(f"Error initializing viewer: {e}")
        raise RuntimeError("Failed to initialize MuJoCo viewer. Make sure mujoco-python-viewer is installed.")

    # --- Control State Initialization & Interactive Loop ---
    current_command = np.zeros(3, dtype=np.float32)
    target_vx, target_vy, target_ang_vel = 0.0, 0.0, 0.0
    rng, loop_key = jax.random.split(rng) # Use loop_key in loop

    print("\n--- Starting Interactive Loop ---")
    print(" Controls: W/S=Vx | A/D=Vy | Q/E=AngularVel | Space=Stop | R=Reset | ESC=Quit")

    run_loop = True
    try:
        while run_loop:
            step_start_time = time.time()
            vx_change, vy_change, ang_vel_change = 0.0, 0.0, 0.0

            # --- Process Key States & Update Velocities ---
            if key_states['W']: vx_change += args.vel_increment
            if key_states['S']: vx_change -= args.vel_increment
            if key_states['A']: vy_change += args.vel_increment
            if key_states['D']: vy_change -= args.vel_increment
            if key_states['Q']: ang_vel_change += args.ang_increment
            if key_states['E']: ang_vel_change -= args.ang_increment

            if key_states['SPACE']:
                target_vx, target_vy, target_ang_vel = 0.0, 0.0, 0.0
                vx_change, vy_change, ang_vel_change = 0.0, 0.0, 0.0
                key_states['SPACE'] = False # Reset trigger

            if key_states['R']:
                print("Resetting environment...")
                loop_key, reset_key = jax.random.split(loop_key)
                state = jit_reset(reset_key)
                mjx_data = state.pipeline_state
                target_vx, target_vy, target_ang_vel = 0.0, 0.0, 0.0
                key_states['R'] = False # Reset trigger

            if key_states['ESC']:
                run_loop = False; continue

            # Apply decay if no key pressed for that axis
            if vx_change == 0.0: target_vx *= args.decay_factor
            else: target_vx += vx_change
            if vy_change == 0.0: target_vy *= args.decay_factor
            else: target_vy += vy_change
            if ang_vel_change == 0.0: target_ang_vel *= args.decay_factor
            else: target_ang_vel += ang_vel_change

            # Clamp velocities
            target_vx = np.clip(target_vx, -args.max_vx, args.max_vx)
            target_vy = np.clip(target_vy, -args.max_vy, args.max_vy)
            target_ang_vel = np.clip(target_ang_vel, -args.max_ang_vel, args.max_ang_vel)

            # --- Update Command & Run Policy ---
            current_command = jp.array([target_vx, target_vy, target_ang_vel])
            if hasattr(state, 'info') and 'command' in state.info:
                 # state = state.replace(info=state.info | {'command': current_command}) # Python 3.9+ merge
                 state = state.replace(info={**state.info, 'command': current_command}) # Compatible merge


            loop_key, act_key = jax.random.split(loop_key)
            try:
                action, _ = jit_inference_fn(state.obs, act_key)
            except Exception as e:
                 print(f"\nError during policy inference: {e}"); traceback.print_exc(); run_loop = False; continue

            # --- Step Environment & Render ---
            state = jit_step(state, action)
            mjx_data = state.pipeline_state

            try:
                viewer.data = mjx_data
                viewer.render()
            except Exception as e:
                 if "window glfw" in str(e).lower() or "context" in str(e).lower(): print("\nViewer closed or context error.")
                 else: print(f"\nViewer render error: {e}.")
                 run_loop = False

            # --- Timing & Done Handling ---
            time_until_next_step = env.dt - (time.time() - step_start_time)
            if time_until_next_step > 0: time.sleep(time_until_next_step)

            if state.done:
                print("\nEpisode done. Resetting...", end=' ')
                loop_key, reset_key = jax.random.split(loop_key)
                state = jit_reset(reset_key)
                mjx_data = state.pipeline_state
                target_vx = target_vy = target_ang_vel = 0.0
                print("Reset complete.")

    except KeyboardInterrupt: print("\nInterrupted.")
    except Exception as e: print(f"\nUnhandled error: {e}"); traceback.print_exc()
    finally:
        print("Cleaning up...")
        try:
             if viewer and viewer.is_alive: viewer.close(); print("Viewer closed.")
        except Exception as e: print(f"Error closing viewer: {e}")
        print("Exiting script.")

if __name__ == "__main__":
    main() 