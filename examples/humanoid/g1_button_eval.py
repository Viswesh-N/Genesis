import argparse
import pickle
import torch
import genesis as gs
from g1_button_env import G1ButtonEnv
import os
os.environ["PYOPENGL_PLATFORM"] = "glx"
os.environ["MUJOCO_GL"] = "glfw" 
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["DISPLAY"] = ":0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--g1_urdf", type=str, default=None, help="Path to G1 URDF file")
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(backend=gs.gpu)

    # Load configurations from training
    log_dir = "/".join(args.model_path.split("/")[:-1])  # get log directory from model path
    
    try:
        # Try loading from new Genesis pattern (single cfgs.pkl file)
        cfgs = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
        env_cfg, reward_cfg, robot_cfg = cfgs[0], cfgs[1], cfgs[2]
        print("Loaded configurations from training logs")
    except FileNotFoundError:
        print("Configuration files not found. Using default configurations.")
        # Use default configurations with new Genesis pattern
        from g1_button_train import get_cfgs
        env_cfg, reward_cfg, robot_cfg = get_cfgs()

    # Override G1 URDF path if provided
    if args.g1_urdf:
        robot_cfg["g1_urdf_path"] = args.g1_urdf

    # Create environment
    env = G1ButtonEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=True,  # Show viewer for evaluation
    )

    # Load trained model
    print(f"Loading model from: {args.model_path}")
    try:
        # Try to load the model state dict
        model_state = torch.load(args.model_path, map_location=gs.device)
        
        # Create policy network (simplified for evaluation)
        from torch import nn
        
        class SimplePolicy(nn.Module):
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
        
        policy = SimplePolicy(env_cfg["num_obs"], env_cfg["num_actions"]).to(gs.device)
        
        # Try to load model weights
        if "model_state_dict" in model_state:
            policy.load_state_dict(model_state["model_state_dict"])
        elif "actor" in model_state:
            policy.load_state_dict(model_state["actor"])
        else:
            policy.load_state_dict(model_state)
            
        policy.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Using random policy for evaluation...")
        policy = None

    # Evaluation loop
    total_episodes = 0
    successful_episodes = 0
    total_reward = 0.0
    
    obs = env.compute_observations()
    
    with torch.no_grad():
        while total_episodes < args.num_episodes:
            if policy is not None:
                actions = policy(obs)
            else:
                # Random actions
                actions = torch.randn(args.num_envs, env_cfg["num_actions"], device=gs.device)
                actions = torch.clamp(actions, -1.0, 1.0)
            
            obs, rewards, dones, info = env.step(actions)
            total_reward += rewards.sum().item()
            
            # Count completed episodes
            num_done = dones.sum().item()
            if num_done > 0:
                # Check which episodes were successful (button pressed)
                button_pos = env.button.get_pos()
                button_displacement = torch.norm(button_pos - env.initial_button_pos, dim=1)
                success_mask = button_displacement > env.button_press_threshold
                
                successful_episodes += (dones & success_mask).sum().item()
                total_episodes += num_done
                
                if total_episodes % 50 == 0:
                    success_rate = successful_episodes / total_episodes
                    avg_reward = total_reward / total_episodes
                    print(f"Episodes: {total_episodes}, Success Rate: {success_rate:.2%}, Avg Reward: {avg_reward:.2f}")

    # Final statistics
    success_rate = successful_episodes / total_episodes
    avg_reward = total_reward / total_episodes
    
    print("\n" + "="*50)
    print(f"EVALUATION COMPLETE")
    print(f"Total Episodes: {total_episodes}")
    print(f"Successful Episodes: {successful_episodes}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Reward: {avg_reward:.2f}")
    print("="*50)


if __name__ == "__main__":
    main()