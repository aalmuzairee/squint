from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence, Union

import dacite
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig

from .robot.so100 import SO100
from .robot.so101 import SO101


@dataclass
class PlaceRandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization config for Place task, extending wrist camera randomization."""
    # Noisy joint positions for better sim2real
    robot_qpos_noise_std: float = np.deg2rad(5)
    # Cube-specific randomization
    cube_half_size_range: Sequence[float] = (0.022 / 2, 0.028 / 2)
    # Can-specific randomization
    can_radius_range: Sequence[float] = (0.028 / 2, 0.038 / 2)
    can_half_height_range: Sequence[float] = (0.05 / 2, 0.07 / 2)
    # Bin randomization (half sizes)
    bin_half_size_x_range: Sequence[float] = (0.07 / 2, 0.09 / 2)
    bin_half_size_y_range: Sequence[float] = (0.09 / 2, 0.11 / 2)
    bin_half_size_z_range: Sequence[float] = (0.024 / 2, 0.036 / 2)

    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False


class Place(DefaultCameraEnv):
    """
    **Task Description:**
    Pick up an item (cube or can) and place it in a bin.

    **Randomizations:**
    - the item's xy position is randomized on top of a table
    - the item's z-axis rotation is randomized
    - the bin's xy position is randomized (non-overlapping with item)

    **Success Conditions:**
    - the item is in the bin xy range
    - the robot is not touching the item or the bin
    - the robot is static
    """

    SUPPORTED_ROBOTS = ["so100", "so101"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation", "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]
    agent: Union[SO100, SO101]

    def __init__(
        self,
        *args,
        item_type="cube",
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[
            PlaceRandomizationConfig, dict
        ] = PlaceRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=[0.3, 0],
        spawn_box_half_size=0.2 / 2,
        **kwargs,
    ):
        self.item_type = item_type

        # Robot-specific configuration
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()

        # Handle domain randomization config
        self.domain_randomization_config = PlaceRandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=PlaceRandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, PlaceRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        self.spawn_box_pos = spawn_box_pos
        self.spawn_box_half_size = spawn_box_half_size

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=True
            if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        if self.item_type not in ["cube", "can"]:
            raise NotImplementedError(f"Unknown item_type: {self.item_type}")

        # Default values
        colors = np.zeros((self.num_envs, 3))
        colors[:, 0] = 1  # Red
        cfg = self.domain_randomization_config
        frictions = np.ones(self.num_envs) * (cfg.item_friction_range[0] + cfg.item_friction_range[1]) / 2
        densities = np.ones(self.num_envs) * (cfg.item_density_range[0] + cfg.item_density_range[1]) / 2

        if self.item_type == "cube":
            half_sizes = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.cube_half_size_range[1]
                    + self.domain_randomization_config.cube_half_size_range[0]
                )
                / 2
            )
            if self.domain_randomization:
                half_sizes = self._batched_episode_rng.uniform(
                    low=cfg.cube_half_size_range[0],
                    high=cfg.cube_half_size_range[1],
                )
                if cfg.randomize_item_color:
                    colors = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
                frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_friction_range[0],
                    high=cfg.item_friction_range[1],
                )
                densities = self._batched_episode_rng.uniform(
                    low=cfg.item_density_range[0],
                    high=cfg.item_density_range[1],
                )
            self.item_half_sizes = common.to_tensor(half_sizes, device=self.device)
            self.item_dimensions = torch.stack([self.item_half_sizes] * 3, dim=-1)

        elif self.item_type == "can":
            colors = np.zeros((self.num_envs, 3))
            colors[:, :] = 0
            colors[:, 2] = 1 # blue
            half_radii = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.can_radius_range[1]
                    + self.domain_randomization_config.can_radius_range[0]
                )
                / 2
            )
            half_heights = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.can_half_height_range[1]
                    + self.domain_randomization_config.can_half_height_range[0]
                )
                / 2
            )
            if self.domain_randomization:
                half_radii = self._batched_episode_rng.uniform(
                    low=cfg.can_radius_range[0],
                    high=cfg.can_radius_range[1],
                )
                half_heights = self._batched_episode_rng.uniform(
                    low=cfg.can_half_height_range[0],
                    high=cfg.can_half_height_range[1],
                )
                if cfg.randomize_item_color:
                    colors = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
                frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_friction_range[0],
                    high=cfg.item_friction_range[1],
                )
                densities = self._batched_episode_rng.uniform(
                    low=cfg.item_density_range[0],
                    high=cfg.item_density_range[1],
                )
            self.item_half_radii = common.to_tensor(half_radii, device=self.device)
            self.item_half_heights = common.to_tensor(half_heights, device=self.device)
            self.item_half_sizes = self.item_half_heights
            self.item_dimensions = torch.stack([self.item_half_radii, self.item_half_radii, self.item_half_heights], dim=-1)

        colors = np.concatenate([colors, np.ones((self.num_envs, 1))], axis=-1)
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        # Build items
        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            friction = frictions[i]
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0,
            )

            if self.item_type == "cube":
                builder.add_box_collision(
                    half_size=[half_sizes[i]] * 3, material=material, density=densities[i]
                )
                builder.add_box_visual(
                    half_size=[half_sizes[i]] * 3,
                    material=sapien.render.RenderMaterial(base_color=colors[i]),
                )
                builder.initial_pose = sapien.Pose(p=[0.2, 0, half_sizes[i]])  # Offset to avoid collision with bin at creation

            elif self.item_type == "can":
                cylinder_pose = sapien.Pose(q=euler2quat(0, np.pi / 2, 0))
                builder.add_cylinder_collision(
                    radius=half_radii[i], half_length=half_heights[i], material=material, density=densities[i],
                    pose=cylinder_pose
                )
                builder.add_cylinder_visual(
                    radius=half_radii[i],
                    half_length=half_heights[i],
                    material=sapien.render.RenderMaterial(base_color=colors[i]),
                    pose=cylinder_pose
                )
                builder.initial_pose = sapien.Pose(p=[0.2, 0, half_heights[i]])  # Offset to avoid collision with bin at creation

            builder.set_scene_idxs([i])
            item = builder.build(name=f"item-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        self.item = Actor.merge(items, name="item")
        self.add_to_state_dict_registry(self.item)

        # Build bins (per-env for domain randomization)
        bin_color = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
        thickness = 0.005
        self.bin_thickness = thickness

        # Default bin half sizes (mid-range)
        cfg = self.domain_randomization_config
        bin_half_sizes_x = np.ones(self.num_envs) * (cfg.bin_half_size_x_range[0] + cfg.bin_half_size_x_range[1]) / 2
        bin_half_sizes_y = np.ones(self.num_envs) * (cfg.bin_half_size_y_range[0] + cfg.bin_half_size_y_range[1]) / 2
        bin_half_sizes_z = np.ones(self.num_envs) * (cfg.bin_half_size_z_range[0] + cfg.bin_half_size_z_range[1]) / 2

        if self.domain_randomization:
            bin_half_sizes_x = self._batched_episode_rng.uniform(
                low=cfg.bin_half_size_x_range[0], high=cfg.bin_half_size_x_range[1]
            )
            bin_half_sizes_y = self._batched_episode_rng.uniform(
                low=cfg.bin_half_size_y_range[0], high=cfg.bin_half_size_y_range[1]
            )
            bin_half_sizes_z = self._batched_episode_rng.uniform(
                low=cfg.bin_half_size_z_range[0], high=cfg.bin_half_size_z_range[1]
            )

        self.bin_half_sizes_x = common.to_tensor(bin_half_sizes_x, device=self.device)
        self.bin_half_sizes_y = common.to_tensor(bin_half_sizes_y, device=self.device)
        self.bin_half_sizes_z = common.to_tensor(bin_half_sizes_z, device=self.device)
        self.bin_dimensions = torch.stack([self.bin_half_sizes_x, self.bin_half_sizes_y, self.bin_half_sizes_z], dim=-1)

        bins = []
        for i in range(self.num_envs):
            bin_half_size = [bin_half_sizes_x[i], bin_half_sizes_y[i], bin_half_sizes_z[i]]
            builder = self.scene.create_actor_builder()

            # Bin floor
            bin_center_pose = sapien.Pose([0.0, 0.0, thickness / 2])
            bin_center_half_size = [bin_half_size[0], bin_half_size[1], thickness / 2]
            builder.add_box_collision(pose=bin_center_pose, half_size=bin_center_half_size)
            builder.add_box_visual(pose=bin_center_pose, half_size=bin_center_half_size, material=bin_color)

            # Bin walls
            for j in [-1, 1]:
                # Y walls
                y = j * bin_center_half_size[1]
                wall_pose = sapien.Pose([0, y, bin_half_size[2]])
                wall_half_size = [bin_half_size[0], thickness / 2, bin_half_size[2]]
                builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)
                # X walls
                x = j * bin_center_half_size[0]
                wall_pose = sapien.Pose([x, 0, bin_half_size[2]])
                wall_half_size = [thickness / 2, bin_half_size[1], bin_half_size[2]]
                builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)

            builder.initial_pose = sapien.Pose(p=[-0.2, 0, bin_half_size[2]])  # Offset to avoid collision with item at creation
            builder.set_scene_idxs([i])
            bin_actor = builder.build(name=f"bin-{i}")
            bins.append(bin_actor)
            self.remove_from_state_dict_registry(bin_actor)

        self.bin = Actor.merge(bins, name="bin")
        self.add_to_state_dict_registry(self.bin)

        self.bin_radius = torch.linalg.norm(self.bin_dimensions[:, :2], dim=-1)

        # Set up greenscreening - keep robot, item, and bin visible
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.item)
            self.remove_object_from_greenscreen(self.bin)

        # Convert rest_qpos to tensor
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        # Table pose
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        # Build camera mount
        self._load_camera_mount()

        # Randomize robot color
        self._randomize_robot_color()

        # Goal site
        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_sphere_visual(
            radius=0.01,
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]),
        )
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            # Random initial qpos
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            # Sample positions for item and bin
            spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )

            # Use placement sampler for non-overlapping positions
            region = [
                [-self.spawn_box_half_size, -self.spawn_box_half_size],
                [self.spawn_box_half_size, self.spawn_box_half_size]
            ]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            # Item/bin radius (use max for conservative placement)
            if self.item_type == "can":
                item_radius = self.item_half_radii.max().item() + 0.01
            else:
                item_radius = self.item_half_sizes.max().item() + 0.01
            bin_radius = self.bin_radius.max().item() + 0.01

            item_xy_offset = sampler.sample(item_radius, 100)
            bin_xy_offset = sampler.sample(bin_radius, 100, verbose=False)

            # Set item pose
            item_xyz = torch.zeros((b, 3))
            item_xyz[:, :2] = spawn_center[env_idx, :2] + item_xy_offset
            item_xyz[:, 2] = self.item_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.item.set_pose(Pose.create_from_pq(item_xyz, qs))

            # Set bin pose
            bin_xyz = torch.zeros((b, 3))
            bin_xyz[:, :2] = spawn_center[env_idx, :2] + bin_xy_offset
            bin_xyz[:, 2] = self.bin_thickness / 2
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.bin.set_pose(Pose.create_from_pq(bin_xyz, qs))

            # Goal is above bin center
            goal_xyz = bin_xyz.clone()
            goal_xyz[:, 2] = self.bin_thickness + self.item_half_sizes[env_idx]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        # Adding joint noise for better sim2real
        if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
            noise = torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
            qpos = qpos + noise
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: dict):
        obs = dict()
        if self.obs_mode_struct.state:
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                is_item_grasped=info["is_item_grasped"],
                item_pose=self.item.pose.raw_pose,
                bin_pose=self.bin.pose.raw_pose,
                tcp_pose=self.agent.tcp_pose.raw_pose,
                tcp_to_item_grip_pos=self.item.pose.p - self.agent.tcp_pos,
                tcp_to_bin_pos=self.bin.pose.p - self.agent.tcp_pos,
                item_to_bin_pos=self.bin.pose.p - self.item.pose.p,
            )
            if self.domain_randomization:
                gripper_params = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    item_dimensions=self.item_dimensions,
                    bin_dimensions=self.bin_dimensions,
                    item_friction=self.item_frictions,
                    item_density=self.item_densities,
                    gripper_stiffness=gripper_params["gripper_stiffness"],
                    gripper_damping=gripper_params["gripper_damping"],
                )
        return obs

    def evaluate(self):
        # Check if item (x,y) position is inside the bin's (x,y) range, in the bin's own frame since the bin spawns with a random yaw. 
        offset = self.item.pose.p - self.bin.pose.p
        w, x, y, z = self.bin.pose.q.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        inside_x = torch.abs(cos_yaw * offset[:, 0] + sin_yaw * offset[:, 1]) < self.bin_half_sizes_x
        inside_y = torch.abs(-sin_yaw * offset[:, 0] + cos_yaw * offset[:, 1]) < self.bin_half_sizes_y
        is_item_above_bin = inside_x & inside_y

        item_lifted = self.item.pose.p[..., -1] >= (self.item_half_sizes + 1e-3)

        item_vel = torch.linalg.norm(self.item.linear_velocity, axis=-1)
        is_item_static = item_vel <= 2e-2
        is_item_grasped = self.agent.is_grasping(self.item)
        is_robot_static = self.agent.is_static()

        # Contact checks
        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_bin = self.agent.is_touching(self.bin)
        robot_touching_item = self.agent.is_touching(self.item)

        success = is_item_above_bin & (~robot_touching_item) & is_robot_static & (~robot_touching_bin)

        return {
            "inside_x": inside_x,
            "inside_y": inside_y,
            "item_vel": item_vel,
            "item_lifted": item_lifted,
            "is_item_static": is_item_static,

            "success": success,
            "is_item_above_bin": is_item_above_bin,
            "is_item_grasped": is_item_grasped,
            "is_robot_static": is_robot_static,
            "robot_touching_table": robot_touching_table,
            "robot_touching_bin": robot_touching_bin,
            "robot_touching_item": robot_touching_item,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Reaching reward
        tcp_to_item_dist = torch.linalg.norm(self.agent.tcp_pose.p - self.item.pose.p, axis=1)
        reaching_reward = 2 * (1 - torch.tanh(5 * tcp_to_item_dist))
        reward = reaching_reward

        # Complex place reward
        item_pos = self.item.pose.p
        bin_pos = self.bin.pose.p.clone()
        goal_xyz = bin_pos.clone()
        goal_xyz[..., 2] = self.bin_thickness + self.item_half_sizes

        # Overall distance reward
        item_to_goal_dist = torch.linalg.norm(goal_xyz - item_pos, axis=1)
        place_reward_final = 1 - torch.tanh(5.0 * item_to_goal_dist)

        # XY and Z distance with far/close logic
        item_to_goal_dist_xy = torch.linalg.norm(goal_xyz[..., :2] - item_pos[..., :2], dim=1)
        # Far: target is above bin (encourages lifting before placing)
        item_to_goal_dist_z_far = torch.linalg.norm(
            (goal_xyz[..., 2:] + (self.bin_dimensions[:, 2:] * 2) + 0.03) - item_pos[..., 2:], dim=1
        )
        # Close: target is final position
        item_to_goal_dist_z_close = torch.linalg.norm(goal_xyz[..., 2:] - item_pos[..., 2:], dim=1)
        item_close_to_goal = (item_to_goal_dist_xy <= self.bin_radius)
        item_to_goal_dist_z = torch.where(item_close_to_goal, item_to_goal_dist_z_close, item_to_goal_dist_z_far)
        place_reward_z = 1 - torch.tanh(10.0 * item_to_goal_dist_z)
        place_reward = place_reward_final + place_reward_z

        # Ungrasp reward (inverted from Reach's close gripper)
        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        gripper_openness = (self.agent.robot.get_qpos()[:, -1] - gripper_min) / (gripper_max - gripper_min)

        # Grasped: 3 + place_reward
        reward[info["is_item_grasped"]] = (3 + place_reward)[info["is_item_grasped"]]

        # Above bin: 3 + place_reward + gripper_openness
        is_item_dropped = (~info["robot_touching_item"]).float()
        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-1], axis=1) 
        static_robot_reward = 1 - torch.tanh(robot_v * 10)
        reward[info["is_item_above_bin"]] = (4 + place_reward + is_item_dropped + gripper_openness + static_robot_reward)[info["is_item_above_bin"]]


        # Success
        reward[info["success"]] = 9

        # Penalties
        reward -= 6 * info["robot_touching_table"].float()
        reward -= 3 * info["robot_touching_bin"].float()
        reward -= 1 * (~info["item_lifted"]).float()  # Encourage picking item fast


        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 9


@register_env("SO101PlaceCube-v1", max_episode_steps=50)
class PlaceCube(Place):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)


@register_env("SO101PlaceCan-v1", max_episode_steps=50)
class PlaceCan(Place):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="can", **kwargs)
