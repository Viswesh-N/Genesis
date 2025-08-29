import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from g1_button_env import G1ButtonEnv


def get_train_cfg(exp_name, max_iterations):
    """Get training configuration for PPO"""
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.02,      
            "gamma": 0.98,             
            "lam": 0.95,
            "learning_rate": 0.0003,   # Lower learning rate for manipulation
            "max_grad_norm": 0.5,      # Smaller gradient clipping
            "num_learning_epochs": 5,   # Fewer epochs per iteration
            "num_mini_batches": 4,     
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 2.0,    
        },
        "init_member_classes": {},
        "policy": {
            "activation": "relu",
            "actor_hidden_dims": [256, 128, 64],     
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 0.3,     # Reduced noise for manipulation
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 64,   # More steps per environment
        "save_interval": 50,
        "empirical_normalization": None,
        "seed": 1,
    }
    
    return train_cfg_dict


def get_cfgs():
    """Get all configurations following Genesis patterns"""
    env_cfg = {
        "num_obs": 50,  # will be adjusted based on robot DOFs
        "num_actions": 20,  # will be adjusted based on robot actuated joints
        "ctrl_dt": 0.02,  # 50 Hz control
        "episode_length_s": 10.0,  
        "action_scales": [0.5] * 20,  
        "target_pos": [0.8, 0.0, 1.0],
        "reach_threshold": 0.1,
    }
    
    reward_scales = {
        "distance": 10.0,        
        "reach_target": 100.0,   
        "action_rate": -0.005,   
        "energy": -0.0005,       
        "stability": -1.0,       
    }
    
    robot_cfg = {
        "g1_urdf_path": "/home/viswesh/grid/ManiSkill/mani_skill/assets/robots/g1_humanoid/g1.urdf",
        "kp": 100.0,  
        "kv": 10.0,   
    }
    
    return env_cfg, reward_scales, robot_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations")
    parser.add_argument("--exp_name", type=str, default="g1_button_push", help="Experiment name")
    parser.add_argument("--headless", action="store_true", help="Run headless (no viewer)")
    parser.add_argument("--g1_urdf", type=str, default=None, help="Path to G1 URDF file")
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(backend=gs.gpu)

    # Get configurations following Genesis patterns
    env_cfg, reward_scales, robot_cfg = get_cfgs()
    
    # Override G1 URDF path if provided
    if args.g1_urdf:
        robot_cfg["g1_urdf_path"] = args.g1_urdf

    print("Creating environment to determine observation and action dimensions...")
    
    # Create a single environment to get dimensions (always headless for dimension detection)
    temp_env = G1ButtonEnv(
        num_envs=1,
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=False,  # Always false for temp env
    )
    
    # Get actual observation and action dimensions
    obs = temp_env.compute_observations()
    
    actual_num_obs = obs.shape[1] if obs.dim() > 1 else obs.shape[0]
    actual_num_actions = temp_env.n_actuated_joints
    
    print(f"Detected {actual_num_obs} observations and {actual_num_actions} actions")
    print(f"Temp env observation shape: {obs.shape}")
    
    # Update configurations with actual dimensions
    env_cfg["num_obs"] = actual_num_obs
    env_cfg["num_actions"] = actual_num_actions
    env_cfg["action_scales"] = [0.3] * actual_num_actions  # Conservative but reasonable movement
    
    # Clean up temp environment
    del temp_env

    # Create the training environment
    print(f"Creating training environment with {args.num_envs} parallel environments...")
    env = G1ButtonEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=not args.headless,
    )

    # Get training configuration
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Create log directory following Genesis patterns
    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Save configurations following Genesis patterns
    pickle.dump(
        [env_cfg, reward_scales, robot_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # Create runner with log_dir parameter following Genesis patterns
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Start training
    print(f"Starting training for {args.max_iterations} iterations...")
    print(f"Logs will be saved to: {log_dir}")
    
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    # Save final model
    final_model_path = f"{log_dir}/model_final.pt"
    runner.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()