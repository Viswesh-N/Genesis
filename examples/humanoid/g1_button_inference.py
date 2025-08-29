import argparse
import pickle
import torch
import numpy as np
import genesis as gs
from g1_button_env import G1ButtonEnv

import os
os.environ["PYOPENGL_PLATFORM"] = "glx"
os.environ["MUJOCO_GL"] = "glfw" 
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["DISPLAY"] = ":0"



def load_model(model_path, num_obs, num_actions, device):
    """Load trained policy model"""
    try:
        # Create policy network
        from torch import nn
        
        class PolicyNetwork(nn.Module):
            def __init__(self, num_obs, num_actions):
                super().__init__()
                self.actor = nn.Sequential(
                    nn.Linear(num_obs, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_actions),
                    nn.Tanh(),
                )
            
            def forward(self, obs):
                return self.actor(obs)
        
        policy = PolicyNetwork(num_obs, num_actions).to(device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Try different checkpoint formats
        if "model_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["model_state_dict"])
        elif "actor_critic" in checkpoint:
            # Extract only actor weights
            actor_weights = {}
            for key, value in checkpoint["actor_critic"].items():
                if key.startswith("actor."):
                    new_key = key.replace("actor.", "")
                    actor_weights[new_key] = value
            policy.actor.load_state_dict(actor_weights)
        else:
            # Handle direct ActorCritic state dict (more common case)
            actor_weights = {}
            for key, value in checkpoint.items():
                if key.startswith("actor."):
                    # Remove "actor." prefix to match our PolicyNetwork.actor structure
                    new_key = key.replace("actor.", "")
                    actor_weights[new_key] = value
            
            if actor_weights:
                # Load only actor weights, ignore critic and std
                policy.actor.load_state_dict(actor_weights)
            else:
                # If no actor prefix, try loading directly (fallback)
                policy.load_state_dict(checkpoint)
            
        policy.eval()
        print(f"Successfully loaded policy from: {model_path}")
        return policy
        
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}: {e}")
        print("Using random policy for demonstration")
        return None


def main():
    parser = argparse.ArgumentParser(description="G1 Target Reaching Inference")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model (optional)")
    parser.add_argument("--g1_urdf", type=str, default=None, help="Path to G1 URDF file")
    parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes to run")
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(backend=gs.gpu)

    # Load configurations if model path is provided
    env_cfg = {
        "num_obs": 62,  # Will be auto-detected
        "num_actions": 27,  # Will be auto-detected
        "ctrl_dt": 0.02,
        "episode_length_s": 15.0,  # Longer episodes for inference
        "action_scales": [0.5] * 27,
    }
    
    reward_cfg = {
        "distance": 10.0,
        "reach": 5.0,
        "reach_target": 20.0,
    }
    
    robot_cfg = {
        "g1_urdf_path": args.g1_urdf or "/home/viswesh/grid/ManiSkill/mani_skill/assets/robots/g1_humanoid/g1.urdf",
        "kp": 100.0,
        "kv": 10.0,
    }
    
    if args.model_path and args.model_path.endswith('.pt'):

        log_dir = "/".join(args.model_path.split("/")[:-1])
        try:
            cfgs = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
            env_cfg, reward_cfg, robot_cfg = cfgs[0], cfgs[1], cfgs[2]
            print("Loaded configuration from training logs")
        except FileNotFoundError:
            print("Using default configuration")

    # Create environment with viewer enabled
    print("Creating inference environment...")
    env = G1ButtonEnv(
        num_envs=1,  # Single environment for inference
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=True,  # Show the viewer for inference
    )

    # Load trained model
    policy = None
    if args.model_path:
        policy = load_model(args.model_path, env_cfg["num_obs"], env_cfg["num_actions"], gs.device)

    print("\n" + "="*60)
    print("G1 TARGET REACHING INFERENCE")
    print("="*60)
    print(f"Episodes to run: {args.num_episodes}")
    print(f"Policy: {'Trained' if policy else 'Random'}")
    print(f"Robot: {'G1 URDF' if args.g1_urdf else 'Default Humanoid'}")
    print(f"Target position: {env.target_pos.tolist()}")
    print("="*60 + "\n")

    # Run inference episodes
    total_rewards = []
    success_count = 0
    
    for episode in range(args.num_episodes):
        print(f"Episode {episode + 1}/{args.num_episodes}")
        
        # Reset environment
        obs, _ = env.reset()  # reset() returns (obs, info)
        
        episode_reward = 0.0
        step_count = 0
        done = False
        
        while not done and step_count < env.max_episode_length:
            # Get action from policy
            with torch.no_grad():
                if policy is not None:
                    action = policy(obs)
                else:
                    # Random actions for demonstration
                    action = torch.randn(1, env.n_actuated_joints, device=gs.device)
                    action = torch.clamp(action, -1.0, 1.0) * 0.3  # Small random movements

            # Step environment
            obs, rewards, dones, _ = env.step(action)
            
            episode_reward += rewards[0].item()
            done = dones[0].item()
            step_count += 1
            
            # Check if target was reached
            ee_pos = env.robot.get_links_pos([env.end_effector_idx]).squeeze()
            target_pos = env.target_pos
            
            distance_to_target = torch.norm(ee_pos - target_pos).item()
            is_success = distance_to_target < env.reach_threshold
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                print(f"  Step {step_count:3d}: Reward={episode_reward:6.2f}, "
                      f"Dist to target={distance_to_target:.3f}m, "
                      f"Target reached={is_success}")
        
        # Episode summary
        total_rewards.append(episode_reward)
        if is_success:
            success_count += 1
            print(f"  SUCCESS! Target reached (distance: {distance_to_target:.3f}m)")
        else:
            print(f"  Failed to reach target (distance: {distance_to_target:.3f}m)")
        
        print(f"  Episode reward: {episode_reward:.2f}, Steps: {step_count}\n")

    # Final statistics
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / args.num_episodes
    
    print("="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"Total Episodes: {args.num_episodes}")
    print(f"Successful Episodes: {success_count}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Reward Range: {min(total_rewards):.2f} - {max(total_rewards):.2f}")
    print("="*60)
    
    if success_rate > 0.8:
        print("Excellent performance!")
    elif success_rate > 0.5:
        print("Good performance!")
    elif success_rate > 0.2:
        print("Room for improvement")
    else:
        print("Needs more training")


if __name__ == "__main__":
    main()