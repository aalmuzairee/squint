"""Base environment classes with domain randomization support.

This module provides a clean hierarchy of environment classes:
- BaseRandomEnv: Common DR (gripper, lighting, robot color) + overlay support
- ThirdCameraEnv: Third-person camera with every-step pose randomization
- WristCameraEnv: Wrist camera with gripper-following randomization

Usage:
    from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig

    class MyTask(DefaultCameraEnv):
        ...
"""

# =============================================================================
# CHANGE THIS TO SWITCH CAMERA TYPE FOR ALL TASKS
# Options: "wrist" or "third"
# =============================================================================
CAMERA_TYPE = "wrist"
# =============================================================================
# This sets the following aliases (defined at bottom of file):
#   "wrist" -> DefaultCameraEnv = WristCameraEnv
#   "third" -> DefaultCameraEnv = ThirdCameraEnv
# DefaultRandomizationConfig = RandomizationConfig (unified config for both)
# =============================================================================

import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import sapien
import torch
from sapien.render import RenderBodyComponent

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.structs import Pose
from mani_skill.utils.visualization.misc import tile_images

from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult


@dataclass
class RandomizationConfig:
    # === Static settings (not affected by domain_randomization flag) ===
    initial_qpos_noise_scale: float = 0.02
    """Noise scale for initial robot joint positions."""
    apply_overlay: bool = True
    """Whether to apply background overlay (greenscreen). If False, returns raw simulation images."""
    rgb_overlay_path: Optional[str] = os.path.join(os.path.dirname(__file__), "black_overlay.png")
    """Path to background image. If None and apply_overlay=True, uses black background."""

    # === Common randomization settings (affected by domain_randomization flag) ===
    gripper_stiffness_range: Sequence[float] = (500, 2000)
    """Range for gripper joint stiffness randomization (per-episode)."""
    gripper_damping_range: Sequence[float] = (50, 200)
    """Range for gripper joint damping randomization (per-episode)."""
    robot_color: Optional[Union[str, Sequence[float]]] = None
    """Robot color in RGB (0-1). Set to "random" for per-episode randomization."""
    randomize_lighting: bool = True
    """Whether to randomize ambient lighting."""

    # === Third-person camera settings (only used by ThirdCameraEnv) ===
    third_camera_pos_noise: Sequence[float] = (0.025, 0.025, 0.025)
    """Max camera position noise from base position (x, y, z)."""
    third_camera_target_noise: float = 0.001
    """Noise scale for camera look-at target position."""
    third_camera_rot_noise: float = np.deg2rad(1)
    """Noise scale for camera view rotation."""
    third_camera_fov_noise: float = np.deg2rad(5)
    """Noise scale for camera FOV."""

    # === Wrist camera settings (only used by WristCameraEnv) ===
    wrist_camera_pos_noise: Sequence[float] = (0.002, 0.002, 0.002)
    """Max position noise (x, y, z) relative to gripper."""
    wrist_camera_rot_noise: Sequence[float] = (np.deg2rad(1), np.deg2rad(1), np.deg2rad(1))
    """Max rotation noise (roll, pitch, yaw) in radians."""
    wrist_camera_fov_noise: float = np.deg2rad(1)
    """Noise scale for camera FOV. Base FOV is 71 degrees."""

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


class BaseRandomEnv(BaseEnv):
    """Base environment with domain randomization and overlay support.

    Handles:
    - Gripper stiffness/damping randomization
    - Lighting randomization
    - Robot color randomization
    - Background overlay (greenscreen) compositing

    Subclasses (ThirdCameraEnv, WristCameraEnv) handle camera-specific logic.
    """

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[RandomizationConfig, dict] = RandomizationConfig(),
        domain_randomization: bool = True,
        **kwargs,
    ):
        self.domain_randomization = domain_randomization

        # Parse config
        self.domain_randomization_config = RandomizationConfig()
        if isinstance(domain_randomization_config, dict):
            merged_config = self.domain_randomization_config.dict()
            common.dict_merge(merged_config, domain_randomization_config)
            for key, value in merged_config.items():
                if hasattr(self.domain_randomization_config, key):
                    setattr(self.domain_randomization_config, key, value)
        elif isinstance(domain_randomization_config, RandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        # Overlay state
        self._objects_to_remove_from_greenscreen: list[Union[Actor, Link]] = []
        self._segmentation_ids_to_keep: torch.Tensor = None
        self._rgb_overlay_image: torch.Tensor = None
        self._overlay_initialized = False

        # Load overlay image as numpy array (will convert to tensor in _after_reconfigure)
        self._rgb_overlay_np = None
        if (
            self.domain_randomization_config.apply_overlay
            and self.domain_randomization_config.rgb_overlay_path is not None
        ):
            path = self.domain_randomization_config.rgb_overlay_path
            if not os.path.exists(path):
                raise FileNotFoundError(f"rgb_overlay_path {path} not found.")
            self._rgb_overlay_np = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        super().__init__(*args, **kwargs)


    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=10)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, 0.3, 0.35], [0.3, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 52 * np.pi / 180, 0.01, 100)

    @property
    def apply_greenscreen(self) -> bool:
        """Backward-compatible alias for apply_overlay."""
        return self.domain_randomization_config.apply_overlay

    def _load_scene(self, options: dict):
        """Initialize scene. Subclasses should call super()._load_scene() first."""
        self._objects_to_remove_from_greenscreen = []

    def _load_lighting(self, options: dict):
        """Load scene lighting with optional randomization."""
        if self.domain_randomization and self.domain_randomization_config.randomize_lighting:
            ambient_colors = self._batched_episode_rng.uniform(0.2, 0.5, size=(3,))
            for i, scene in enumerate(self.scene.sub_scenes):
                scene.render_system.ambient_light = ambient_colors[i]
        else:
            self.scene.set_ambient_light([0.3, 0.3, 0.3])

        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=False, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_camera_mount(self):
        """Create camera mount actors for pose randomization."""
        # Third-person camera mount
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.camera_mount = builder.build_kinematic("camera_mount")

        # Wrist camera mount
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.wrist_camera_mount = builder.build_kinematic("wrist_camera_mount")

    def _randomize_robot_color(self):
        """Apply robot color randomization if configured."""
        if self.domain_randomization_config.robot_color is None:
            return

        for link in self.agent.robot.links:
            for i, obj in enumerate(link._objs):
                render_body_component: RenderBodyComponent = obj.entity.find_component_by_type(
                    RenderBodyComponent
                )
                if render_body_component is None:
                    continue

                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        if (
                            self.domain_randomization
                            and self.domain_randomization_config.robot_color == "random"
                        ):
                            color = self._batched_episode_rng[i].uniform(0.0, 1.0, size=(3,)).tolist()
                        else:
                            color = list(self.domain_randomization_config.robot_color)
                        part.material.set_base_color(color + [1])

    def _randomize_gripper_speed(self, env_idx: torch.Tensor):
        """Randomize gripper stiffness/damping per episode."""
        stiff_lo, stiff_hi = self.domain_randomization_config.gripper_stiffness_range
        damp_lo, damp_hi = self.domain_randomization_config.gripper_damping_range

        # Initialize storage for privileged observations
        if not hasattr(self, "_gripper_stiffness"):
            default_stiffness = (stiff_lo + stiff_hi) / 2
            default_damping = (damp_lo + damp_hi) / 2
            self._gripper_stiffness = torch.full((self.num_envs,), default_stiffness, device=self.device)
            self._gripper_damping = torch.full((self.num_envs,), default_damping, device=self.device)

        if not self.domain_randomization:
            return
        if stiff_lo == stiff_hi and damp_lo == damp_hi:
            return

        stiffnesses = self._batched_episode_rng[env_idx].uniform(stiff_lo, stiff_hi)
        dampings = self._batched_episode_rng[env_idx].uniform(damp_lo, damp_hi)
        gripper_joint = self.agent.robot.joints_map["gripper"]

        for i, idx in enumerate(env_idx.tolist()):
            gripper_joint._objs[idx].set_drive_properties(stiffnesses[i], dampings[i], force_limit=100)
            self._gripper_stiffness[idx] = stiffnesses[i]
            self._gripper_damping[idx] = dampings[i]

    def get_gripper_params(self) -> dict[str, torch.Tensor]:
        """Get normalized gripper parameters for privileged observations."""
        stiff_lo, stiff_hi = self.domain_randomization_config.gripper_stiffness_range
        damp_lo, damp_hi = self.domain_randomization_config.gripper_damping_range

        stiff_range = stiff_hi - stiff_lo if stiff_hi != stiff_lo else 1.0
        damp_range = damp_hi - damp_lo if damp_hi != damp_lo else 1.0

        return {
            "gripper_stiffness": (self._gripper_stiffness - stiff_lo) / stiff_range,
            "gripper_damping": (self._gripper_damping - damp_lo) / damp_range,
        }

    def remove_object_from_greenscreen(self, obj: Union[Articulation, Actor, Link]):
        """Mark an object to be kept in foreground (not replaced by overlay)."""
        if isinstance(obj, Articulation):
            for link in obj.get_links():
                self._objects_to_remove_from_greenscreen.append(link)
        elif isinstance(obj, (Actor, Link)):
            self._objects_to_remove_from_greenscreen.append(obj)

    def _after_reconfigure(self, options: dict):
        """Build segmentation IDs and load overlay image to GPU."""
        super()._after_reconfigure(options)

        if not self.domain_randomization_config.apply_overlay:
            self._objects_to_remove_from_greenscreen = []
            return

        # Build segmentation IDs to keep
        per_scene_ids = []
        for obj in self._objects_to_remove_from_greenscreen:
            per_scene_ids.append(obj.per_scene_id)

        if per_scene_ids:
            self._segmentation_ids_to_keep = torch.unique(torch.concatenate(per_scene_ids))
        else:
            self._segmentation_ids_to_keep = torch.tensor([], dtype=torch.int64)

        # Load overlay image to GPU
        if not self._overlay_initialized and self._rgb_overlay_np is not None:
            # Get camera resolution from sensor config
            for name, sensor in self._sensor_configs.items():
                if isinstance(sensor, CameraConfig) and name != "render_camera":
                    # Resize to camera resolution
                    resized = cv2.resize(self._rgb_overlay_np, (sensor.width, sensor.height))
                    self._rgb_overlay_image = common.to_tensor(resized, device=self.device)
                    break

            # If no camera found, use default size
            if self._rgb_overlay_image is None and self._rgb_overlay_np is not None:
                self._rgb_overlay_image = common.to_tensor(self._rgb_overlay_np, device=self.device)

        # Create black overlay if no image provided but overlay is enabled
        if not self._overlay_initialized and self._rgb_overlay_image is None:
            for name, sensor in self._sensor_configs.items():
                if isinstance(sensor, CameraConfig) and name != "render_camera":
                    self._rgb_overlay_image = torch.zeros(
                        (sensor.height, sensor.width, 3), dtype=torch.uint8, device=self.device
                    )
                    break

        self._overlay_initialized = True
        self._objects_to_remove_from_greenscreen = []

    def _green_screen_rgb(self, rgb: torch.Tensor, segmentation: torch.Tensor, overlay: torch.Tensor) -> torch.Tensor:
        """Apply background overlay using segmentation mask."""
        actor_seg = segmentation[..., 0]
        mask = torch.ones_like(actor_seg, dtype=torch.bool)

        if self._segmentation_ids_to_keep.device != actor_seg.device:
            self._segmentation_ids_to_keep = self._segmentation_ids_to_keep.to(actor_seg.device)

        # Keep foreground objects (robot, task objects)
        mask[torch.isin(actor_seg, self._segmentation_ids_to_keep)] = False
        mask = mask[..., None]

        # Composite: foreground where mask=False, overlay where mask=True
        original_dtype = rgb.dtype
        rgb = rgb.float()
        overlay = overlay.float()
        result = rgb * (~mask) + overlay * mask

        return result.to(original_dtype)

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
        """Get sensor observations with optional overlay applied."""
        obs = super()._get_obs_sensor_data(apply_texture_transforms)

        if not self.domain_randomization_config.apply_overlay:
            return obs

        if not (self.obs_mode_struct.visual.rgb and self.obs_mode_struct.visual.segmentation):
            return obs

        if self._rgb_overlay_image is None:
            return obs

        # Apply overlay to all RGB cameras
        for camera_name, camera_obs in obs.items():
            if not isinstance(camera_obs, dict) or "rgb" not in camera_obs:
                continue
            if "segmentation" not in camera_obs:
                continue
            if camera_name == "render_camera":
                continue

            overlay = self._rgb_overlay_image
            if overlay.device != camera_obs["rgb"].device:
                self._rgb_overlay_image = overlay.to(camera_obs["rgb"].device)
                overlay = self._rgb_overlay_image

            obs[camera_name]["rgb"] = self._green_screen_rgb(
                camera_obs["rgb"],
                camera_obs["segmentation"],
                overlay,
            )

        return obs


    def render_all(self):
        """Renders all human render cameras and sensors together, excluding segmentation."""

        images = []
        for obj in self._hidden_objects:
            obj.show_visual()
        self.scene.update_render(update_sensors=True, update_human_render_cameras=True)
        render_images = self.scene.get_human_render_camera_images()
        sensor_images = self.get_sensor_images()

        # Render sensor first and then human renders
        for image in sensor_images.values():
            for key, img in image.items():
                # Skip segmentation images
                if "segmentation" not in key:
                    images.append(img)
        for image in render_images.values():
            images.append(image)

        return tile_images(images)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Base episode initialization. Subclasses should call super() first."""
        self._randomize_gripper_speed(env_idx)


class ThirdCameraEnv(BaseRandomEnv):
    """Environment with third-person camera and every-step pose randomization.

    Camera pose is randomized at every control step when domain_randomization=True.
    """

    # Default camera position and target
    DEFAULT_CAMERA_POS = [0.6, 0.3, 0.3]
    DEFAULT_CAMERA_TARGET = [0.3, 0, 0.05]
    DEFAULT_CAMERA_FOV = np.deg2rad(60)  # 60 degrees

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[RandomizationConfig, dict] = RandomizationConfig(),
        **kwargs,
    ):
        self.base_camera_settings = dict(
            pos=self.DEFAULT_CAMERA_POS,
            target=self.DEFAULT_CAMERA_TARGET,
        )

        super().__init__(*args, domain_randomization_config=domain_randomization_config, **kwargs)

    @property
    def _default_sensor_configs(self):
        config = self.domain_randomization_config

        # FOV randomization
        if self.domain_randomization and config.third_camera_fov_noise > 0:
            fov_noise = config.third_camera_fov_noise * (2 * self._batched_episode_rng.rand() - 1)
        else:
            fov_noise = 0

        return [
            CameraConfig(
                "base_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=self.DEFAULT_CAMERA_FOV + fov_noise,
                near=0.01,
                far=100,
                mount=self.camera_mount,
            )
        ]

    def sample_camera_poses(self, n: int):
        """Sample randomized camera poses."""
        from mani_skill.utils.structs import Pose

        config = self.domain_randomization_config

        if not self.domain_randomization:
            # Return static pose
            static_pose = sapien_utils.look_at(
                eye=self.base_camera_settings["pos"],
                target=self.base_camera_settings["target"],
            )
            # raw_pose may have shape [1, 1, 7] or [1, 7], squeeze to [7] then expand to [n, 7]
            pose_tensor = static_pose.raw_pose.squeeze()
            return Pose.create(pose_tensor.unsqueeze(0).expand(n, -1))

        # Convert to tensors if needed
        pos = common.to_tensor(self.base_camera_settings["pos"], device=self.device)
        target = common.to_tensor(self.base_camera_settings["target"], device=self.device)
        max_offset = common.to_tensor(config.third_camera_pos_noise, device=self.device)

        # Sample random eye positions
        eyes = randomization.camera.make_camera_rectangular_prism(
            n,
            scale=max_offset,
            center=pos,
            theta=0,
            device=self.device,
        )

        # Sample poses with noise
        poses = randomization.camera.noised_look_at(
            eyes,
            target=target,
            look_at_noise=config.third_camera_target_noise,
            view_axis_rot_noise=config.third_camera_rot_noise,
            device=self.device,
        )

        return poses

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized camera pose."""
        super()._initialize_episode(env_idx, options)
        self.camera_mount.set_pose(self.sample_camera_poses(n=len(env_idx)))

    def _before_control_step(self):
        """Randomize camera pose every step."""
        if self.domain_randomization:
            self.camera_mount.set_pose(self.sample_camera_poses(n=self.num_envs))
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()



class WristCameraEnv(BaseRandomEnv):
    """Environment with wrist camera that follows gripper with randomization.

    Camera is mounted relative to gripper_link and follows gripper movement.
    Position and rotation offsets are randomized every step when domain_randomization=True.
    """

    # Base pose relative to gripper_link
    WRIST_CAMERA_BASE_POS = (-0.0049, 0.0498, -0.0591)
    WRIST_CAMERA_BASE_ROT_RAD = (np.deg2rad(-90), np.deg2rad(91), np.deg2rad(-35.31))  # radians (roll, pitch, yaw)
    WRIST_CAMERA_FOV = np.deg2rad(71)  # 71 degrees

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[RandomizationConfig, dict] = RandomizationConfig(),
        **kwargs,
    ):
        super().__init__(*args, domain_randomization_config=domain_randomization_config, **kwargs)

    @property
    def _default_sensor_configs(self):
        config = self.domain_randomization_config

        # FOV noise (randomized per-env at initialization)
        if self.domain_randomization and config.wrist_camera_fov_noise > 0:
            fov_noise = config.wrist_camera_fov_noise * (2 * self._batched_episode_rng.rand() - 1)
        else:
            fov_noise = 0

        return [
            CameraConfig(
                "base_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=self.WRIST_CAMERA_FOV + fov_noise,
                near=0.01,
                far=100,
                mount=self.wrist_camera_mount,
            )
        ]

    def _update_wrist_camera_pose(self):
        """Update wrist camera mount to follow gripper with random offsets."""
        config = self.domain_randomization_config
        gripper_pose = self.agent.robot.links_map["gripper_link"].pose

        base_x, base_y, base_z = self.WRIST_CAMERA_BASE_POS
        base_roll, base_pitch, base_yaw = self.WRIST_CAMERA_BASE_ROT_RAD

        if self.domain_randomization:
            # Batch all random numbers into one call (6 values per env)
            rand_vals = 2 * torch.rand(self.num_envs, 6, device=self.device) - 1

            pos_offset = config.wrist_camera_pos_noise
            rot_noise = config.wrist_camera_rot_noise

            dx = pos_offset[0] * rand_vals[:, 0]
            dy = pos_offset[1] * rand_vals[:, 1]
            dz = pos_offset[2] * rand_vals[:, 2]
            d_roll = rot_noise[0] * rand_vals[:, 3]
            d_pitch = rot_noise[1] * rand_vals[:, 4]
            d_yaw = rot_noise[2] * rand_vals[:, 5]
        else:
            dx = dy = dz = torch.zeros(self.num_envs, device=self.device)
            d_roll = d_pitch = d_yaw = torch.zeros(self.num_envs, device=self.device)

        # Final position and rotation
        px, py, pz = base_x + dx, base_y + dy, base_z + dz
        roll_rad, pitch_rad, yaw_rad = base_roll + d_roll, base_pitch + d_pitch, base_yaw + d_yaw

        # Convert euler to quaternion (batched)
        cj, sj = torch.cos(pitch_rad / 2), torch.sin(pitch_rad / 2)
        ck, sk = torch.cos(yaw_rad / 2), torch.sin(yaw_rad / 2)
        ci, si = torch.cos(roll_rad / 2), torch.sin(roll_rad / 2)

        q_py_w, q_py_x, q_py_y, q_py_z = cj * ck, sj * sk, sj * ck, cj * sk

        qw = q_py_w * ci - q_py_x * si
        qx = q_py_w * si + q_py_x * ci
        qy = q_py_y * ci + q_py_z * si
        qz = q_py_z * ci - q_py_y * si

        p = torch.stack([px, py, pz], dim=-1)
        q = torch.stack([qw, qx, qy, qz], dim=-1)

        local_offset = Pose.create_from_pq(p=p, q=q)
        self.wrist_camera_mount.set_pose(gripper_pose * local_offset)

    def reset(self, *args, **kwargs):
        """Reset and sync wrist camera for correct first frame."""
        obs, info = super().reset(*args, **kwargs)
        # Sync wrist camera pose once at reset for correct first frame
        # Parent reset ends with _gpu_apply_all, so we need fetch first
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self._update_wrist_camera_pose()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene._gpu_fetch_all()  # Complete the cycle
        # The parent's observation was rendered before the camera moved; observe again from the synced pose
        obs = self.get_obs(info)
        self._last_obs = obs
        return obs, info

    def _after_control_step(self):
        """Update wrist camera pose after physics step."""
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self._update_wrist_camera_pose()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()


# =============================================================================
# Default aliases based on CAMERA_TYPE setting at top of file
# =============================================================================
if CAMERA_TYPE == "wrist":
    DefaultCameraEnv = WristCameraEnv
elif CAMERA_TYPE == "third":
    DefaultCameraEnv = ThirdCameraEnv
else:
    raise ValueError(f"Unknown CAMERA_TYPE: {CAMERA_TYPE}. Use 'wrist' or 'third'")

DefaultRandomizationConfig = RandomizationConfig
