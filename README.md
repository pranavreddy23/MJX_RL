# MJX Reinforcement Learning for Locomotion

<!-- Optional: Add a relevant image or GIF here -->
<!-- <p align="center">
  <a href="[Link to video if available]"><img src="[Link to image/gif]" alt="[Description]" style="width:800px"/></a>
</p> -->

This project focuses on training reinforcement learning (PPO) agents for simulated quadruped and humanoid locomotion tasks using the [Brax](https://github.com/google/brax) library, leveraging the [MJX](https://mujoco.readthedocs.io/en/latest/mjx.html) physics engine within MuJoCo.

<!-- Optional: Add links to relevant papers if this code is associated with publications -->
<!--
Code related to the papers:
- [Paper Title 1](Link)
- [Paper Title 2](Link)
-->

## Project Structure

A rough outline for the repository:
├── checkpoints/ # Default location for saved model checkpoints
│ ├── quadruped/ # Checkpoints for the quadruped environment
│ └── humanoid/ # Checkpoints for the humanoid environment
├── configs/ # Configuration files (e.g., PPO hyperparameters)
│ └── default_configs.py
├── environments/ # Environment definitions (observation/action spaces, rewards, reset, step)
│ ├── init.py
│ ├── quadruped.py
│ └── humanoid.py
├── humanoid/ # Humanoid MJCF model files (XMLs/meshes/textures)
│ └── humanoid.xml # Example model file
├── policies/ # Policy training implementations (PPO logic, network creation)
│ ├── init.py
│ └── ppo.py
├── utils/ # Utility scripts (e.g., rendering, domain randomization)
│ ├── init.py
│ ├── domain_rand.py
│ └── rendering.py
├── main.py # Main script for training and


## Setup / Requirements

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Virtual Environment (Recommended):**
    ```bash
    python -m venv mjx_env
    source mjx_env/bin/activate  # On Linux/macOS
    # .\mjx_env\Scripts\activate  # On Windows
    ```
3.  **Install Dependencies:**
    Ensure you have a compatible version of MuJoCo installed (>3.1.0 recommended for MJX). Install the necessary Python packages using pip. Key dependencies include:
    *   `jax` (consider GPU version, e.g., `jax[cuda12_pip]`)
    *   `brax`
    *   `mujoco`
    *   `flax`
    *   `orbax-checkpoint`
    *   `mediapy`
    *   `ml_collections`
    *   `etils[epath]`

    Example installation command:
    ```bash
    pip install "jax[cuda12_pip]" brax mujoco flax orbax-checkpoint mediapy ml_collections "etils[epath]"
    ```
    *(Adjust JAX version based on your CUDA setup if applicable. Verify Brax compatibility.)*

## Usage

The main entry point is `main.py`.

#### **To Train:**

*   **Train Quadruped (default):**
    ```bash
    python main.py --env quadruped
    ```
*   **Train Humanoid:**
    ```bash
    python main.py --env humanoid
    ```
*   **Specify Checkpoint Directory:**
    ```bash
    python main.py --env quadruped --brax_checkpoint_dir /path/to/my/checkpoints
    ```
*   **Resume Training:** Training automatically resumes from the latest valid checkpoint (or `final_model`) found in the environment's subdirectory within `--brax_checkpoint_dir` (e.g., `./checkpoints/quadruped/`).
*   **Resume from Specific Checkpoint:**
    ```bash
    python main.py --env quadruped --load_checkpoint ./checkpoints/quadruped/000010000000
    ```

#### **To Evaluate (Play Policy):**

*   **Evaluate Latest Quadruped Policy:**
    ```bash
    python main.py --env quadruped --eval_only
    ```
*   **Evaluate Specific Humanoid Checkpoint (using final_model link):**
    ```bash
    python main.py --env humanoid --eval_only --load_checkpoint ./checkpoints/humanoid/final_model
    ```
*   **Evaluate Specific Humanoid Checkpoint (using step number):**
    ```bash
    python main.py --env humanoid --eval_only --load_checkpoint ./checkpoints/humanoid/000020000000
    ```

#### **Rendering:**

Rendering is enabled by default after training or evaluation (`--render`).
*   A video (`policy_render.mp4`) will be saved.
*   An HTML visualization (`policy_visualization.html`) will be saved.
Both files are placed in the environment's checkpoint directory (e.g., `./checkpoints/quadruped/`).

#### **Command-line Arguments:**

Use `python main.py --help` to see all available options.

## Checkpoints

*   Checkpoints are saved within the directory specified by `--brax_checkpoint_dir`, organized into subdirectories by environment name (e.g., `./checkpoints/quadruped/`).
*   After successful training, a symlink named `final_model` is created within the environment's checkpoint directory, pointing to the latest saved checkpoint step (e.g., `./checkpoints/quadruped/final_model` -> `./checkpoints/quadruped/000020000000`). This `final_model` path is prioritized when loading checkpoints automatically.

<!-- Optional: Add Citation section if applicable -->
<!--
## Citation
If you find this work useful in your own research, please consider citing:
```bibtex
@misc{your_project_2024,
  author = {Your Name(s)},
  title = {MJX Reinforcement Learning for Locomotion},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{<your-repo-url>}},
}
```
-->

<!-- Optional: Add Credits section if applicable -->
<!--
### Credits
This code structure may be inspired by other repositories like X, Y, Z.
-->