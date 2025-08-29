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

        self.target_pos = torch.tensor([0.8, 0.0, 1.0], device=self.device)  # Target position
        self.target_pos_list = [0.8, 0.0, 1.0]  # as list for morph creation
        self.reach_threshold = 0.1  # Success when within 10cm

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
                pos=(0.9, 0, 0.8),  # Lower wall to match new button height
                size=(0.05, 2.0, 1.6),  # Make wall shorter since robot is lower
                fixed=True,
            )
        )

        # Simple red target sphere (no physics, just visual)
        self.target = self.scene.add_entity(
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
            morph=gs.morphs.Sphere(
                pos=self.target_pos_list,
                radius=0.05,
                fixed=True,
                collision=False,  # No collision, just visual target
            )
        )

        g1_urdf_path = robot_cfg.get("g1_urdf_path", "/home/viswesh/grid/ManiSkill/mani_skill/assets/robots/g1_humanoid/g1.urdf")
        
        self.robot = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=g1_urdf_path,
                pos=(0, 0, 0.85),  # Position so feet touch the ground
                fixed=False,  # Allow robot to move
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
            kp[frozen_joint_idx] = 50000.0  # Even higher position gain for stability
            kv[frozen_joint_idx] = 2000.0   # Even higher velocity damping
        
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

        # Target is fixed, no need to track initial position
        
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

        # Target is fixed, no need to reset position

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
        # Clip actions to prevent extreme values (following Go2 pattern)
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Store last actions for observation
        all_joint_pos = self.robot.get_dofs_position()
        if self.num_envs == 1:
            if all_joint_pos.dim() == 1:
                self.last_actions = all_joint_pos[self.actuated_joints].unsqueeze(0)
            else:
                self.last_actions = all_joint_pos[:, self.actuated_joints]
        else:
            self.last_actions = all_joint_pos[:, self.actuated_joints]
        
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
        
        # Continuously set frozen joints to standing position
        if len(self.frozen_joints) > 0:
            standing_positions = [
                -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,  # Left leg
                -0.1, 0.0, 0.0, 0.2, -0.1, 0.0   # Right leg
            ]
            if len(standing_positions) >= len(self.frozen_joints):
                frozen_pos_tensor = torch.tensor(standing_positions[:len(self.frozen_joints)], device=self.device)
                frozen_pos_tensor = frozen_pos_tensor.unsqueeze(0).repeat(self.num_envs, 1)
                self.robot.control_dofs_position(frozen_pos_tensor, self.frozen_joints)

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
        # robot joint positions and velocities (only actuated joints)
        all_joint_pos = self.robot.get_dofs_position()
        all_joint_vel = self.robot.get_dofs_velocity()
        
        # Extract only actuated joints
        if self.num_envs == 1:
            if all_joint_pos.dim() == 1:
                joint_pos = all_joint_pos[self.actuated_joints].unsqueeze(0)
                joint_vel = all_joint_vel[self.actuated_joints].unsqueeze(0)
            else:
                joint_pos = all_joint_pos[:, self.actuated_joints]
                joint_vel = all_joint_vel[:, self.actuated_joints]
        else:
            joint_pos = all_joint_pos[:, self.actuated_joints]
            joint_vel = all_joint_vel[:, self.actuated_joints]

        # Get robot base orientation and compute projected gravity (like Go2)
        base_quat = self.robot.get_quat()  # (num_envs, 4) or (4,)
        if self.num_envs == 1 and base_quat.dim() == 1:
            base_quat = base_quat.unsqueeze(0)
        
        # Compute projected gravity vector
        global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0)
        if self.num_envs > 1:
            global_gravity = global_gravity.repeat(self.num_envs, 1)
            
        from genesis.utils.geom import transform_by_quat, inv_quat
        inv_base_quat = inv_quat(base_quat)
        projected_gravity = transform_by_quat(global_gravity, inv_base_quat)

        # end effector position
        ee_pos = self.robot.get_links_pos([self.end_effector_idx])
        if self.num_envs == 1:
            ee_pos = ee_pos.squeeze()
            if ee_pos.dim() == 1:
                ee_pos = ee_pos.unsqueeze(0)
        else:
            ee_pos = ee_pos.squeeze(1)
        
        # target position (fixed)
        target_pos = self.target_pos
        if self.num_envs == 1:
            target_pos = target_pos.unsqueeze(0)
        else:
            target_pos = target_pos.unsqueeze(0).repeat(self.num_envs, 1)

        # relative position (ee to target) - more important than absolute positions
        rel_pos = target_pos - ee_pos
        
        # distance to target
        distance_to_target = torch.norm(rel_pos, dim=1, keepdim=True)

        # previous action (for action smoothness)
        if not hasattr(self, 'last_actions'):
            self.last_actions = torch.zeros_like(joint_pos[:, :self.n_actuated_joints])
        
        # Apply observation scaling (following Go2 pattern)
        joint_pos_scaled = joint_pos * 1.0  # joint positions don't need heavy scaling
        joint_vel_scaled = joint_vel * 0.05  # scale down velocities
        rel_pos_scaled = rel_pos * 2.0  # emphasize relative position
        distance_scaled = distance_to_target * 5.0  # emphasize distance

        # concatenate observations (following successful patterns)
        obs = torch.cat([
            projected_gravity,      # 3 - orientation info (like Go2)
            joint_pos_scaled,       # n_joints - joint positions 
            joint_vel_scaled,       # n_joints - joint velocities
            rel_pos_scaled,         # 3 - relative position to target (most important)
            distance_scaled,        # 1 - distance to target
            self.last_actions,      # n_joints - previous actions
        ], dim=1)

        return obs

    # ------------ reward functions following Genesis patterns ----------------
    def _reward_distance(self):
        """Dense distance reward with exponential decay"""
        ee_pos = self.robot.get_links_pos([self.end_effector_idx])
        target_pos = self.target_pos
        
        # handle single vs multiple environment cases
        if self.num_envs == 1:
            ee_pos = ee_pos.squeeze(0)
            distance_to_target = torch.norm(ee_pos - target_pos)
            distance_to_target = distance_to_target.unsqueeze(0)  # make (1,)
        else:
            ee_pos = ee_pos.squeeze(1)
            distance_to_target = torch.norm(ee_pos - target_pos, dim=1)
        
        # Linear distance reward with better gradient
        # Negative distance + small constant to keep positive
        max_distance = 3.0  # Maximum expected distance
        normalized_distance = torch.clamp(distance_to_target / max_distance, 0.0, 1.0)
        return (1.0 - normalized_distance) * 2.0  # Scale to [0, 2] range
    
    def _reward_reach_target(self):
        """Sparse bonus reward for reaching the target"""
        ee_pos = self.robot.get_links_pos([self.end_effector_idx])
        target_pos = self.target_pos
        
        # handle single vs multiple environment cases
        if self.num_envs == 1:
            ee_pos = ee_pos.squeeze(0)
            distance_to_target = torch.norm(ee_pos - target_pos)
            distance_to_target = distance_to_target.unsqueeze(0)  # make (1,)
        else:
            ee_pos = ee_pos.squeeze(1)
            distance_to_target = torch.norm(ee_pos - target_pos, dim=1)
        
        # Scaled bonus for getting close to target
        close_bonus = torch.clamp((self.reach_threshold - distance_to_target) / self.reach_threshold, 0.0, 1.0)
        target_reached = (distance_to_target < self.reach_threshold).float()
        return close_bonus * 5.0 + target_reached * 20.0  # Progressive bonus + big final bonus
        
    def _reward_action_rate(self):
        """Penalize large changes in actions (smoothness)"""
        if not hasattr(self, 'last_actions'):
            return torch.zeros(self.num_envs, device=self.device)
        
        # Get current actions
        all_joint_pos = self.robot.get_dofs_position()
        if self.num_envs == 1:
            if all_joint_pos.dim() == 1:
                current_joint_pos = all_joint_pos[self.actuated_joints].unsqueeze(0)
            else:
                current_joint_pos = all_joint_pos[:, self.actuated_joints]
        else:
            current_joint_pos = all_joint_pos[:, self.actuated_joints]
        
        # Penalize rapid changes
        action_diff = torch.sum(torch.square(current_joint_pos - self.last_actions), dim=1)
        return action_diff
        
    def _reward_energy(self):
        """Penalize high joint velocities (energy consumption)"""
        all_joint_vel = self.robot.get_dofs_velocity()
        if self.num_envs == 1:
            if all_joint_vel.dim() == 1:
                joint_vel = all_joint_vel[self.actuated_joints].unsqueeze(0)
            else:
                joint_vel = all_joint_vel[:, self.actuated_joints]
        else:
            joint_vel = all_joint_vel[:, self.actuated_joints]
        
        # Penalize high velocities
        return torch.sum(torch.square(joint_vel), dim=1)
    
    def _reward_stability(self):
        """Penalize excessive base movement/tilting"""
        base_pos = self.robot.get_pos()
        base_quat = self.robot.get_quat()
        
        # Handle single vs multiple environment cases
        if self.num_envs == 1:
            if base_pos.dim() == 1:
                base_pos = base_pos.unsqueeze(0)
            if base_quat.dim() == 1:
                base_quat = base_quat.unsqueeze(0)
        
        # Penalize base movement in X and Y (should stay roughly in place)
        base_xy_movement = torch.sum(torch.square(base_pos[:, :2]), dim=1)
        
        # Penalize excessive tilting (quaternion should stay close to upright)
        from genesis.utils.geom import quat_to_xyz
        base_euler = quat_to_xyz(base_quat, rpy=True, degrees=False)
        tilt_penalty = torch.sum(torch.square(base_euler[:, :2]), dim=1)  # Roll and pitch
        
        return base_xy_movement + tilt_penalty * 5.0  # Weight tilt more heavily

    def compute_dones(self):
        """Compute done flags for all environments"""
        # episode timeout
        timeout = self.episode_length_buf >= self.max_episode_length
        
        # Optional: task success (hand reaches target) - but let's disable for now
        # ee_pos = self.robot.get_links_pos([self.end_effector_idx])
        # if self.num_envs == 1:
        #     ee_pos = ee_pos.squeeze(0)
        #     distance_to_target = torch.norm(ee_pos - self.target_pos)
        #     success = distance_to_target < self.reach_threshold
        #     success = success.unsqueeze(0)
        # else:
        #     ee_pos = ee_pos.squeeze(1)
        #     distance_to_target = torch.norm(ee_pos - self.target_pos, dim=1)
        #     success = distance_to_target < self.reach_threshold
        
        # For now, only terminate on timeout to let robot learn
        success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # done if timeout (or success when enabled)
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