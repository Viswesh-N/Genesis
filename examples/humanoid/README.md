# G1 Humanoid Button Push Task

This directory contains a reinforcement learning environment for training the Unitree G1 humanoid robot to push a button on a wall. The task focuses on upper body control while the robot base is fixed for stability.

## Task Description

- **Objective**: Train the G1 robot to reach and push a red button mounted on a wall
- **Robot**: Unitree G1 (or fallback humanoid) with fixed base
- **Environment**: Simple room with a wall and button at position (0.8, 0.0, 1.2)
- **Success**: Button must be displaced by more than 2cm to be considered "pressed"

## Files

- `g1_button_env.py`: Main environment implementation
- `g1_button_train.py`: Training script using PPO via rsl-rl
- `g1_button_eval.py`: Batch evaluation script for trained models
- `g1_button_inference.py`: Interactive inference script with viewer
- `README.md`: This documentation

## Environment Features

### State Space
- Robot joint positions and velocities
- End effector (hand) position
- Button position (target)
- Distance to button
- Button displacement from initial position

### Action Space
- Continuous joint position commands for actuated joints
- Actions are scaled and applied as position targets

### Reward Function
- **Distance reward**: Exponential decay based on distance to button
- **Reach reward**: Bonus when end effector is very close (<10cm)
- **Press reward**: Large reward (10x) when button is successfully pressed

## Setup and Usage

### Prerequisites
```bash
pip install rsl-rl-lib==2.2.4
```

### Getting a G1 URDF
You'll need a Unitree G1 URDF file. You can:
1. Download from Unitree's official resources
2. Use community-provided URDFs from repositories like:
   - [Unitree Robot Models](https://github.com/unitreerobotics/unitree_ros)
   - Other robotics model collections

### Training
```bash
cd examples/humanoid

# With G1 URDF (recommended)
python g1_button_train.py --g1_urdf /path/to/g1.urdf --num_envs 4096 --max_iterations 1000

# Without G1 URDF (uses default humanoid)
python g1_button_train.py --num_envs 4096 --max_iterations 1000

# Headless training (faster)
python g1_button_train.py --headless --num_envs 8192
```

### Evaluation

#### Interactive Inference (with popup window)
```bash
# Run inference with viewer (like grasp/go2 examples)
python g1_button_inference.py --model_path logs/g1_button_push/model_final.pt --num_episodes 5

# With specific G1 URDF
python g1_button_inference.py --model_path logs/g1_button_push/model_final.pt --g1_urdf /path/to/g1.urdf

# Random policy demonstration (no trained model)
python g1_button_inference.py --num_episodes 3
```

#### Batch Evaluation (headless)
```bash
# Evaluate trained model in batch
python g1_button_eval.py --model_path logs/g1_button_push/model_final.pt --num_episodes 100

# With specific G1 URDF
python g1_button_eval.py --model_path logs/g1_button_push/model_final.pt --g1_urdf /path/to/g1.urdf
```

## Configuration

### Environment Parameters
- `ctrl_dt`: 0.02s (50Hz control frequency)
- `episode_length_s`: 10s maximum episode length
- `button_press_threshold`: 0.02m required displacement

### Training Parameters
- Algorithm: PPO with adaptive KL scheduling
- Network: [512, 256, 128] hidden layers
- Learning rate: 0.0003
- Entropy coefficient: 0.01 (for exploration)

### Reward Weights
- Distance reward: 1.0
- Reach reward: 2.0
- Press reward: 20.0

## Customization

### Modifying the Task
- Change button position in `g1_button_env.py` (`self.button_pos`)
- Adjust reward weights in `get_reward_cfg()`
- Modify episode length or control frequency in `get_env_cfg()`

### Adding More Complexity
- Multiple buttons
- Moving targets
- Obstacles in the environment
- Balance requirements (unfix the robot base)

## Expected Results

With proper training:
- Robot should learn to extend its arm toward the button
- Success rate should increase over training iterations
- Final policy should achieve >90% success rate on button pressing

## Troubleshooting

### Common Issues
1. **No G1 URDF**: Environment will fall back to default humanoid XML
2. **Joint limits**: Ensure URDF has proper joint limits defined
3. **End effector detection**: Environment tries to auto-detect hand link, may need manual adjustment
4. **Action scaling**: Conservative scaling (0.5) is used by default, may need tuning

### Performance Tips
- Use more environments (8192+) for faster training
- Enable headless mode for better performance
- Monitor success rate and adjust reward weights if needed

## Notes

- Robot base is fixed for stability as requested
- Environment focuses on upper body manipulation
- Button press detection is based on displacement, not force
- State includes button position for easier learning (privileged information)