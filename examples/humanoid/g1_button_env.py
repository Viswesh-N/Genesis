import torch
import math
import genesis as gs


class G1ButtonEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        reward_cfg,
        robot_cfg,
        show_viewer=False,
    ):
        self.num_envs = num_envs
        self.num_obs = env_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        self.button_pos = torch.tensor([0.8, 0.0, 1.2], device=self.device)
        self.button_press_threshold = 0.02
        self.button_pos_list = [0.8, 0.0, 1.2]  # as list for morph creation

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(-2.0, 3.0, 2.5),  # Further back and higher to see full robot
                camera_lookat=(0.0, 0.0, 0.8),  # Look at robot center (around chest height)
                camera_fov=50,  # Slightly wider field of view
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, num_envs)))),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())

        self.wall = self.scene.add_entity(
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
            morph=gs.morphs.Box(
                pos=(0.9, 0, 1.0),
                size=(0.05, 2.0, 2.0),
                fixed=True,
            )
        )

        self.button = self.scene.add_entity(
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
            morph=gs.morphs.Cylinder(
                pos=self.button_pos_list,
                radius=0.03,
                height=0.05,
                quat=(0.7071, 0, 0.7071, 0),
            )
        )

        g1_urdf_path = robot_cfg.get("g1_urdf_path", "/home/viswesh/grid/ManiSkill/mani_skill/assets/robots/g1_humanoid/g1.urdf")
        
        self.robot = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=g1_urdf_path,
                pos=(0, 0, 1.0),  # Raise robot so legs are visible
                fixed=True,
            )
        )
        print(f"Using G1 URDF: {g1_urdf_path}")

        # build the scene
        self.scene.build(n_envs=num_envs)

        # get robot info
        self.robot_dofs = self.robot.n_dofs
        self.robot_links = self.robot.n_links
        
        # Define upper body joints only (freeze lower body)
        upper_body_joint_names = [
            "torso_joint",
            # Left arm
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint", "left_elbow_roll_joint",
            # Right arm  
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint", "right_elbow_roll_joint",
            # Left hand
            "left_zero_joint", "left_one_joint", "left_two_joint", "left_three_joint", 
            "left_four_joint", "left_five_joint", "left_six_joint",
            # Right hand
            "right_zero_joint", "right_one_joint", "right_two_joint", "right_three_joint",
            "right_four_joint", "right_five_joint", "right_six_joint",
        ]
        
        # Get DOF indices for upper body joints only
        self.actuated_joints = []
        for joint_name in upper_body_joint_names:
            try:
                joint = self.robot.get_joint(joint_name)
                if joint.dof_start is not None:
                    if joint.dof_start < self.robot_dofs:  # Ensure valid index
                        self.actuated_joints.append(joint.dof_start)
                    else:
                        print(f"Warning: Joint {joint_name} has DOF index {joint.dof_start} >= {self.robot_dofs}")
            except Exception as e:
                print(f"Warning: Could not find joint {joint_name}: {e}")
                
        self.n_actuated_joints = len(self.actuated_joints)
        print(f"Robot has {self.robot_dofs} total DOFs, using {self.n_actuated_joints} upper body joints")
        print(f"Actuated joints (DOF indices): {self.actuated_joints}")
        
        # Freeze lower body joints by setting their positions to 0 and high damping
        lower_body_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", 
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
        
        frozen_joints = []
        for joint_name in lower_body_joint_names:
            try:
                joint = self.robot.get_joint(joint_name)
                if joint.dof_start is not None:
                    if joint.dof_start < self.robot_dofs:  # Ensure valid index
                        frozen_joints.append(joint.dof_start)
                    else:
                        print(f"Warning: Joint {joint_name} has DOF index {joint.dof_start} >= {self.robot_dofs}")
            except Exception as e:
                print(f"Warning: Could not find joint {joint_name} to freeze: {e}")
        
        print(f"Freezing {len(frozen_joints)} lower body joints: {frozen_joints}")

        # determine end effector link (assume it's the rightmost hand link)
        self.end_effector_idx = self._find_end_effector_link()
        print(f"Using link {self.end_effector_idx} as end effector")

        # episode state
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.rewards = torch.zeros(self.num_envs, device=self.device)
        
        # set control gains from robot_cfg
        kp = torch.ones(self.robot_dofs, device=self.device) * robot_cfg["kp"]
        kv = torch.ones(self.robot_dofs, device=self.device) * robot_cfg["kv"]
        
        # Set very high damping for frozen joints to keep them still
        for frozen_joint_idx in frozen_joints:
            kp[frozen_joint_idx] = 10000.0  # Very high position gain
            kv[frozen_joint_idx] = 1000.0   # Very high velocity damping
        
        self.robot.set_dofs_kp(kp)
        self.robot.set_dofs_kv(kv)
        
        # Store frozen joints for resetting
        self.frozen_joints = frozen_joints

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
            
        # Initialize extras for logging
        self.extras = dict()
        self.extras["observations"] = dict()

        # initial button positions
        initial_pos = self.button.get_pos()  # This could be (3,) or (num_envs, 3)
        if initial_pos.dim() == 1:
            # Single environment: (3,) -> (1, 3)
            self.initial_button_pos = initial_pos.unsqueeze(0)
        else:
            # Multi environment: already (num_envs, 3)
            self.initial_button_pos = initial_pos.clone()
        
        # reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def _find_end_effector_link(self):
        """Find the end effector link (right hand) based on link names or positions"""
        # Try to find a link with 'right' and 'hand' in the name
        for i, link in enumerate(self.robot.links):
            link_name = getattr(link, 'name', f'link_{i}').lower()
            if 'right' in link_name and ('hand' in link_name or 'wrist' in link_name):
                return i
        
        # If no named hand found, use the last few links (common for end effector)
        if self.robot_links > 10:
            return self.robot_links - 3  # often hand/wrist is near the end
        else:
            return self.robot_links - 1  # use last link

    def reset_idx(self, envs_idx):
        """Reset specified environments following Genesis patterns"""
        if len(envs_idx) == 0:
            return
        
        n_resets = len(envs_idx)

        # reset robot joint positions for actuated joints only
        actuated_joint_pos = torch.zeros(n_resets, self.n_actuated_joints, device=self.device)
        actuated_joint_vel = torch.zeros(n_resets, self.n_actuated_joints, device=self.device)
        
        # Add small random variations to all actuated joints
        actuated_joint_pos += torch.randn(n_resets, self.n_actuated_joints, device=self.device) * 0.1
        
        # set actuated joint states
        if self.num_envs == 1:
            # For single environment, don't pass env_ids
            self.robot.set_dofs_position(actuated_joint_pos[0], self.actuated_joints)
            self.robot.set_dofs_velocity(actuated_joint_vel[0], self.actuated_joints)
        else:
            self.robot.set_dofs_position(actuated_joint_pos, self.actuated_joints, envs_idx)
            self.robot.set_dofs_velocity(actuated_joint_vel, self.actuated_joints, envs_idx)
            
        # Reset frozen joints to natural standing position with 0 velocity
        if len(self.frozen_joints) > 0:
            # Set leg joints to natural standing positions instead of all zeros
            # Order: left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
            #        right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll
            standing_positions = [
                -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,  # Left leg (slight bend at hip and knee)
                -0.1, 0.0, 0.0, 0.2, -0.1, 0.0   # Right leg (slight bend at hip and knee)
            ]
            
            # Ensure we have the right number of positions
            if len(standing_positions) >= len(self.frozen_joints):
                frozen_positions = torch.tensor(standing_positions[:len(self.frozen_joints)], device=self.device)
                frozen_positions = frozen_positions.unsqueeze(0).repeat(n_resets, 1)
            else:
                # Fallback to zeros if mismatch
                frozen_positions = torch.zeros(n_resets, len(self.frozen_joints), device=self.device)
                
            frozen_velocities = torch.zeros(n_resets, len(self.frozen_joints), device=self.device)
            if self.num_envs == 1:
                self.robot.set_dofs_position(frozen_positions[0], self.frozen_joints)
                self.robot.set_dofs_velocity(frozen_velocities[0], self.frozen_joints)
            else:
                self.robot.set_dofs_position(frozen_positions, self.frozen_joints, envs_idx)
                self.robot.set_dofs_velocity(frozen_velocities, self.frozen_joints, envs_idx)

        # reset button position
        if self.num_envs == 1:
            # For single environment, use the stored position directly
            self.button.set_pos(self.initial_button_pos[0])
        else:
            # For multiple environments, reset selected environments
            button_pos = self.initial_button_pos[envs_idx]
            self.button.set_pos(button_pos, envs_idx)

        # reset episode tracking
        self.episode_length_buf[envs_idx] = 0
        
        # fill extras for logging (following Genesis patterns)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
    
    def reset(self):
        """Reset all environments following Genesis patterns"""
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        obs, self.extras = self.get_observations()
        return obs, None

    def step(self, actions):
        """Step the environment"""
        # scale and apply actions
        scaled_actions = actions * self.action_scales
        
        # apply actions to actuated joints only (upper body)
        if len(self.actuated_joints) == actions.shape[1]:
            self.robot.control_dofs_position(scaled_actions, self.actuated_joints)
        else:
            # if action dimension doesn't match, use available actions for available joints
            n_actions = min(actions.shape[1], len(self.actuated_joints))
            selected_joints = self.actuated_joints[:n_actions]
            self.robot.control_dofs_position(scaled_actions[:, :n_actions], selected_joints)
        
        # Continuously set frozen joints to 0 position with 0 velocity target
        frozen_positions = torch.zeros(self.num_envs, len(self.frozen_joints), device=self.device)
        if len(self.frozen_joints) > 0:
            self.robot.control_dofs_position(frozen_positions, self.frozen_joints)

        # step simulation
        self.scene.step()

        # update episode tracking
        self.episode_length_buf += 1

        # compute rewards using reward functions (following Genesis patterns)
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            rewards += rew
            self.episode_sums[name] += rew
        
        # compute dones and reset
        dones = self.compute_dones()
        reset_env_ids = torch.where(dones)[0]
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        # compute observations and fill extras
        obs, self.extras = self.get_observations()
        
        return obs, rewards, dones, self.extras

    def compute_observations(self):
        """Compute observations for all environments"""
        # robot joint positions and velocities
        joint_pos = self.robot.get_dofs_position()  # (num_envs, n_dofs) or (n_dofs,)
        joint_vel = self.robot.get_dofs_velocity()  # (num_envs, n_dofs) or (n_dofs,)

        # end effector position
        ee_pos = self.robot.get_links_pos([self.end_effector_idx])  # (num_envs, 1, 3) or (1, 3)
        
        # button position (target)
        button_pos = self.button.get_pos()  # (num_envs, 3) or (3,)

        # handle single vs multiple environment cases
        if self.num_envs == 1:
            # ensure all are 2D for consistency (1, N)
            if joint_pos.dim() == 1:
                joint_pos = joint_pos.unsqueeze(0)  # (n_dofs,) -> (1, n_dofs)
            if joint_vel.dim() == 1:
                joint_vel = joint_vel.unsqueeze(0)  # (n_dofs,) -> (1, n_dofs)
            
            # handle end effector position
            ee_pos = ee_pos.squeeze()  # remove all singleton dims
            if ee_pos.dim() == 1:
                ee_pos = ee_pos.unsqueeze(0)  # (3,) -> (1, 3)
                
            # handle button position
            if button_pos.dim() == 1:
                button_pos = button_pos.unsqueeze(0)  # (3,) -> (1, 3)
                
            initial_button_pos = self.initial_button_pos  # already (1, 3)
        else:
            ee_pos = ee_pos.squeeze(1)  # (num_envs, 1, 3) -> (num_envs, 3)
            # For multi-env, initial_button_pos should be (num_envs, 3)
            if self.initial_button_pos.shape[0] == 1:
                initial_button_pos = self.initial_button_pos.repeat(self.num_envs, 1)
            else:
                initial_button_pos = self.initial_button_pos

        # distance to button
        distance_to_button = torch.norm(ee_pos - button_pos, dim=1, keepdim=True)  # (num_envs, 1)

        # button displacement from initial position
        button_displacement = torch.norm(button_pos - initial_button_pos, dim=1, keepdim=True)  # (num_envs, 1)


        # concatenate observations
        obs = torch.cat([
            joint_pos,      # robot joint positions
            joint_vel,      # robot joint velocities  
            ee_pos,         # end effector position
            button_pos,     # button position (target)
            distance_to_button,    # distance to button
            button_displacement,   # how much button has moved
        ], dim=1)

        return obs

    # ------------ reward functions following Genesis patterns ----------------
    def _reward_distance(self):
        """Distance reward (closer is better)"""
        ee_pos = self.robot.get_links_pos([self.end_effector_idx])
        button_pos = self.button.get_pos()
        
        # handle single vs multiple environment cases
        if self.num_envs == 1:
            ee_pos = ee_pos.squeeze(0)
            distance_to_button = torch.norm(ee_pos - button_pos)
            distance_to_button = distance_to_button.unsqueeze(0)  # make (1,)
        else:
            ee_pos = ee_pos.squeeze(1)
            distance_to_button = torch.norm(ee_pos - button_pos, dim=1)
        
        return torch.exp(-distance_to_button * 5.0)
    
    def _reward_reach(self):
        """Reaching reward (bonus when very close)"""
        ee_pos = self.robot.get_links_pos([self.end_effector_idx])
        button_pos = self.button.get_pos()
        
        # handle single vs multiple environment cases
        if self.num_envs == 1:
            ee_pos = ee_pos.squeeze(0)
            distance_to_button = torch.norm(ee_pos - button_pos)
            distance_to_button = distance_to_button.unsqueeze(0)  # make (1,)
        else:
            ee_pos = ee_pos.squeeze(1)
            distance_to_button = torch.norm(ee_pos - button_pos, dim=1)
        
        return (distance_to_button < 0.1).float() * 0.5
    
    def _reward_press(self):
        """Button press reward (big bonus for actually pressing)"""
        button_pos = self.button.get_pos()
        
        if self.num_envs == 1:
            button_displacement = torch.norm(button_pos - self.initial_button_pos)
            button_displacement = button_displacement.unsqueeze(0)  # make (1,)
        else:
            if self.initial_button_pos.shape[0] == 1:
                initial_button_pos = self.initial_button_pos.repeat(self.num_envs, 1)
            else:
                initial_button_pos = self.initial_button_pos
            button_displacement = torch.norm(button_pos - initial_button_pos, dim=1)
        
        button_pressed = (button_displacement > self.button_press_threshold).float()
        return button_pressed * 10.0

    def compute_dones(self):
        """Compute done flags for all environments"""
        # episode timeout
        timeout = self.episode_length_buf >= self.max_episode_length
        
        # task success (button pressed)
        button_pos = self.button.get_pos()
        if self.num_envs == 1:
            button_displacement = torch.norm(button_pos - self.initial_button_pos)  # scalar
            button_displacement = button_displacement.unsqueeze(0)  # make it (1,)
        else:
            # Handle initial_button_pos for multi-env dones
            if self.initial_button_pos.shape[0] == 1:
                initial_button_pos = self.initial_button_pos.repeat(self.num_envs, 1)
            else:
                initial_button_pos = self.initial_button_pos
            button_displacement = torch.norm(button_pos - initial_button_pos, dim=1)
        
        success = button_displacement > self.button_press_threshold
        
        # done if timeout or success
        done = timeout | success
        
        return done

    def get_states(self):
        """Get privileged states (same as observations for now)"""
        return self.compute_observations()
    
    def get_observations(self):
        """Get observations for RSL-RL compatibility"""
        obs = self.compute_observations()
        extras = {"observations": {}}  # empty observations dict in extras
        return obs, extras