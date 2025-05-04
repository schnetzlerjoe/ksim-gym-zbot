"""Defines simple task for training a walking policy for the default humanoid."""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self, Sequence, TypedDict, Union, cast

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import optax
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from ksim.actuators import NoiseType, StatefulActuators
from ksim.types import PhysicsData
from mujoco import mjx
from mujoco_scenes.mjcf import load_mjmodel

from ksim.utils.mujoco import get_ctrl_data_idx_by_name, log_joint_config

logger = logging.getLogger(__name__)

NUM_JOINTS = 20
NUM_ACTOR_INPUTS = 43
NUM_CRITIC_INPUTS = 444

# These are in the order of the neural network outputs.
ZEROS: list[tuple[str, float]] = [
    ("left_shoulder_pitch", 0.0),
    ("left_shoulder_roll", 0.0),
    ("left_elbow_roll", 0.0),
    ("left_gripper_roll", 0.0),
    ("right_shoulder_pitch", 0.0),
    ("right_shoulder_roll", 0.0),
    ("right_elbow_roll", 0.0),
    ("right_gripper_roll", 0.0),
    ("left_hip_pitch", 0.0),
    ("left_hip_roll", 0.0),
    ("left_hip_yaw", 0.0),
    ("left_knee_pitch", 0.0),
    ("left_ankle_pitch", 0.0),
    ("left_ankle_roll", 0.0),
    ("right_hip_pitch", 0.0),
    ("right_hip_roll", 0.0),
    ("right_hip_yaw", 0.0),
    ("right_knee_pitch", 0.0),
    ("right_ankle_pitch", 0.0),
    ("right_ankle_roll", 0.0),
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PlannerState:
    position: Array
    velocity: Array


@attrs.define(frozen=True, kw_only=True)
class BentArmPenalty(ksim.Reward):
    arm_indices: tuple[int, ...] = attrs.field()
    arm_targets: tuple[float, ...] = attrs.field()

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        qpos = trajectory.qpos[..., self.arm_indices]
        qpos_targets = jnp.array(self.arm_targets)
        qpos_diff = qpos - qpos_targets
        return xax.get_norm(qpos_diff, "l1").mean(axis=-1)

    @classmethod
    def create(
        cls,
        model: ksim.PhysicsModel,
        scale: float,
        scale_by_curriculum: bool = False,
    ) -> Self:
        qpos_mapping = ksim.get_qpos_data_idxs_by_name(model)

        names = [
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_elbow_roll",
            "right_gripper_roll",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_elbow_roll",
            "left_gripper_roll",
        ]

        zeros = {k: v for k, v in ZEROS}
        arm_indices = [qpos_mapping[name][0] for name in names]
        arm_targets = [zeros[name] for name in names]

        return cls(
            arm_indices=tuple(arm_indices),
            arm_targets=tuple(arm_targets),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    num_mixtures: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Reshape the output to be a mixture of gaussians.
        slice_len = NUM_JOINTS * self.num_mixtures
        mean_nm = out_n[..., :slice_len].reshape(NUM_JOINTS, self.num_mixtures)
        std_nm = out_n[..., slice_len : slice_len * 2].reshape(NUM_JOINTS, self.num_mixtures)
        logits_nm = out_n[..., slice_len * 2 :].reshape(NUM_JOINTS, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means.
        mean_nm = mean_nm + jnp.array([v for _, v in ZEROS])[:, None]

        dist_n = ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)

        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_inputs = NUM_CRITIC_INPUTS
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        self.actor = Actor(
            key,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=0.5,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            depth=depth,
        )
        self.critic = Critic(
            key,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class ZbotWalkingTaskConfig(ksim.PPOConfig):
    """Config for the Z-Bot walking task."""

    robot_mjcf_path: str = xax.field(
        value="kscale-assets/zbot-6dof-feet/",
        help="The path to the assets directory for the robot.",
    )

    actuator_params_path: str = xax.field(
        value="kscale-assets/actuators/",
        help="The path to the assets directory for actuator models",
    )

    # Model parameters.
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    scale: float = xax.field(
        value=0.1,
        help="The maximum position delta on each step, in radians.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=2.0,
        help="Maximum gradient norm for clipping.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )

    # Curriculum parameters.
    num_curriculum_levels: int = xax.field(
        value=10,
        help="The number of curriculum levels to use.",
    )
    increase_threshold: float = xax.field(
        value=3.0,
        help="Increase the curriculum level when the mean trajectory length is above this threshold.",
    )
    decrease_threshold: float = xax.field(
        value=1.0,
        help="Decrease the curriculum level when the mean trajectory length is below this threshold.",
    )
    min_level_steps: int = xax.field(
        value=50,
        help="The minimum number of steps to wait before changing the curriculum level.",
    )
    min_curriculum_level: float = xax.field(
        value=0.0,
        help="The minimum curriculum level to use.",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=0,
        help="The body id to track with the render camera.",
    )


class FeetechParams(TypedDict):
    sysid: str
    max_torque: float
    armature: float
    frictionloss: float
    damping: float
    vin: float
    kt: float
    R: float
    vmax: float
    amax: float
    max_velocity: float
    max_pwm: float
    error_gain: float


def trapezoidal_step(
    state: PlannerState, target_position: Array, dt: float
) -> tuple[PlannerState, tuple[Array, Array]]:
    v_max = 5.0
    a_max = 39.0

    position_error = target_position - state.position
    direction = jnp.sign(position_error)

    stopping_distance = (state.velocity**2) / (2 * a_max)

    # Decide accelerate or decelerate
    should_accelerate = jnp.abs(position_error) > stopping_distance

    acceleration = jnp.where(should_accelerate, direction * a_max, -direction * a_max)
    new_velocity = state.velocity + acceleration * dt

    # Clamp velocity
    new_velocity = jnp.clip(new_velocity, -v_max, v_max)

    # Prevent overshoot when decelerating
    new_velocity = jnp.where(direction * new_velocity < 0, 0.0, new_velocity)

    new_position = state.position + new_velocity * dt

    new_state = PlannerState(position=new_position, velocity=new_velocity)

    return new_state, (new_position, new_velocity)


def load_actuator_params(params_path: str, actuator_type: str) -> FeetechParams:
    params_file = Path(params_path) / f"{actuator_type}.json"
    if not params_file.exists():
        raise ValueError(
            f"Actuator parameters file '{params_file}' not found. Please ensure it exists in '{params_path}'."
        )
    with open(params_file, "r") as f:
        return json.load(f)


class FeetechActuators(StatefulActuators):
    """Feetech actuator controller."""

    def __init__(
        self,
        max_torque_j: Array,
        kp_j: Array,
        kd_j: Array,
        max_velocity_j: Array,
        max_pwm_j: Array,
        vin_j: Array,
        kt_j: Array,
        r_j: Array,
        vmax_j: Array,
        amax_j: Array,
        error_gain_j: Array,
        dt: float,
        action_noise: float = 0.0,
        action_noise_type: NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: NoiseType = "none",
    ) -> None:
        self.max_torque_j = max_torque_j
        self.kp_j = kp_j
        self.kd_j = kd_j
        self.max_velocity_j = max_velocity_j
        self.max_pwm_j = max_pwm_j
        self.vin_j = vin_j
        self.kt_j = kt_j
        self.r_j = r_j
        self.vmax_j = vmax_j
        self.amax_j = amax_j
        self.error_gain_j = error_gain_j
        self.dt = dt
        # self.prev_qtarget_j = jnp.zeros_like(self.kp_j)
        self.action_noise = action_noise
        self.action_noise_type = action_noise_type
        self.torque_noise = torque_noise
        self.torque_noise_type = torque_noise_type

    def get_stateful_ctrl(
        self,
        action: Array,
        physics_data: PhysicsData,
        actuator_state: PlannerState,
        rng: PRNGKeyArray,
    ) -> tuple[Array, PlannerState]:
        """Compute torque control with velocity smoothing and duty cycle clipping (JAX friendly)."""
        pos_rng, tor_rng = jax.random.split(rng)

        current_pos_j = physics_data.qpos[7:]
        current_vel_j = physics_data.qvel[6:]

        planner_state = actuator_state
        planner_state, (desired_position, desired_velocity) = trapezoidal_step(planner_state, action, self.dt)

        pos_error_j = desired_position - current_pos_j
        vel_error_j = desired_velocity - current_vel_j

        # Compute raw duty cycle and clip by max_pwm
        raw_duty_j = self.kp_j * self.error_gain_j * pos_error_j + self.kd_j * vel_error_j
        duty_j = jnp.clip(raw_duty_j, -self.max_pwm_j, self.max_pwm_j)

        # Compute torque
        volts_j = duty_j * self.vin_j
        torque_j = volts_j * self.kt_j / self.r_j

        # Add noise to torque
        torque_j_noisy = self.add_noise(self.torque_noise, self.torque_noise_type, torque_j, tor_rng)

        return torque_j_noisy, planner_state

    def get_default_action(self, physics_data: PhysicsData) -> Array:
        return physics_data.qpos[7:]

    def get_default_state(self, initial_position: Array, initial_velocity: Array) -> PlannerState:
        """Initialize the planner state with the provided position and velocity."""
        return PlannerState(position=initial_position, velocity=initial_velocity)

    def get_initial_state(self, physics_data: PhysicsData, rng: PRNGKeyArray) -> PlannerState:
        """Implement abstract method to initialize planner state from physics data."""
        initial_position = physics_data.qpos[7:]
        initial_velocity = physics_data.qvel[6:]
        return self.get_default_state(initial_position, initial_velocity)


class ZbotWalkingTask(ksim.PPOTask[ZbotWalkingTaskConfig]):
    def get_optimizer(self) -> optax.GradientTransformation:
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            (
                optax.adam(self.config.learning_rate)
                if self.config.adam_weight_decay == 0.0
                else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
            ),
        )

        return optimizer

    def _configure_actuator_params(
        self,
        mj_model: mujoco.MjModel,
        dof_id: int,
        joint_name: str,
        params: FeetechParams,
    ) -> None:
        """Configure actuator parameters for a joint."""
        mj_model.dof_damping[dof_id] = params["damping"]
        mj_model.dof_armature[dof_id] = params["armature"]
        mj_model.dof_frictionloss[dof_id] = params["frictionloss"]

        # breakpoint()
        actuator_name = f"{joint_name}_ctrl"
        actuator_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id >= 0:
            # Set force limits flag on the joint (using joint index) rather than actuator index
            joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                mj_model.jnt_actfrclimited[joint_id] = 1
            max_torque = float(params["max_torque"])
            mj_model.actuator_forcerange[actuator_id, :] = [-max_torque, max_torque]

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = (Path(self.config.robot_mjcf_path) / "robot.mjcf").resolve().as_posix()
        logger.info("Loading MJCF model from %s", mjcf_path)
        mj_model = load_mjmodel(mjcf_path, scene="smooth")

        metadata = self.get_mujoco_model_metadata(mj_model)

        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        # Validate parameters
        required_keys = ["damping", "armature", "frictionloss", "max_torque"]

        # Apply servo-specific parameters based on joint metadata
        for i in range(mj_model.njnt):
            joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None:
                logger.warning("Joint at index %d has no name; skipping parameter assignment.", i)
                continue

            # Look up joint metadata. Warn if actuator_type is missing.
            if joint_name not in metadata:
                logger.warning("Joint '%s' is missing; skipping parameter assignment.", joint_name)
                continue

            joint_meta = metadata[joint_name]
            if joint_meta.actuator_type is None:
                logger.warning("Joint '%s' is missing an actuator_type; skipping parameter assignment.", joint_name)
                continue

            dof_id = mj_model.jnt_dofadr[i]

            # Load and validate parameters for this actuator type
            params = load_actuator_params(self.config.actuator_params_path, joint_meta.actuator_type)
            for key in required_keys:
                if key not in params:
                    raise ValueError(f"Missing required key '{key}' in {joint_meta.actuator_type} parameters.")

            # Apply parameters based on the joint suffix
            self._configure_actuator_params(mj_model, dof_id, joint_name, params)

        return mj_model

    # def log_joint_config(self, model: Union[mujoco.MjModel, mjx.Model]) -> None:
    #     metadata = self.get_mujoco_model_metadata(model)
    #     debug_lines = ["==== Joint and Actuator Properties ===="]

    #     if isinstance(model, mujoco.MjModel):
    #         logger.info("******** PhysicsModel is Mujoco")

    #         njnt = model.njnt

    #         def get_joint_name(idx: int) -> Optional[str]:
    #             return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)

    #         def get_actuator_id(name: str) -> int:
    #             return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    #         dof_damping = model.dof_damping
    #         dof_armature = model.dof_armature
    #         dof_frictionloss = model.dof_frictionloss
    #         jnt_dofadr = model.jnt_dofadr
    #         actuator_forcerange = model.actuator_forcerange

    #     elif isinstance(model, mjx.Model):
    #         logger.info("******** PhysicsModel is MJX")

    #         njnt = model.njnt
    #         dof_damping = model.dof_damping
    #         dof_armature = model.dof_armature
    #         dof_frictionloss = model.dof_frictionloss
    #         jnt_dofadr = model.jnt_dofadr
    #         actuator_forcerange = model.actuator_forcerange

    #         def extract_name(byte_array: bytes, adr_array: Sequence[int], idx: int) -> Optional[str]:
    #             adr = adr_array[idx]
    #             if adr < 0:
    #                 return None
    #             end = byte_array.find(b"\x00", adr)
    #             return byte_array[adr:end].decode("utf-8")

    #         actuator_name_to_id = {
    #             extract_name(model.names, model.name_actuatoradr, i): i
    #             for i in range(model.nu)
    #             if model.name_actuatoradr[i] >= 0
    #         }

    #         def get_joint_name(idx: int) -> Optional[str]:
    #             return extract_name(model.names, model.name_jntadr, idx)

    #         def get_actuator_id(name: str) -> int:
    #             return actuator_name_to_id.get(name, -1)

    #     else:
    #         raise TypeError("Unsupported model type provided")

    #     for i in range(njnt):
    #         joint_name = get_joint_name(i)
    #         if joint_name is None:
    #             continue

    #         joint_meta = metadata.get(joint_name)
    #         if not joint_meta:
    #             logger.warning("Joint '%s' missing metadata; skipping.", joint_name)
    #             continue

    #         actuator_type = joint_meta.actuator_type
    #         if actuator_type is None:
    #             logger.warning("Joint '%s' missing actuator_type; skipping.", joint_name)
    #             continue

    #         dof_id = jnt_dofadr[i]
    #         damping = dof_damping[dof_id]
    #         armature = dof_armature[dof_id]
    #         frictionloss = dof_frictionloss[dof_id]
    #         joint_id = joint_meta.id if joint_meta.id is not None else "N/A"
    #         kp = joint_meta.kp if joint_meta.kp is not None else "N/A"
    #         kd = joint_meta.kd if joint_meta.kd is not None else "N/A"

    #         actuator_name = f"{joint_name}_ctrl"
    #         actuator_id = get_actuator_id(actuator_name)

    #         line = (
    #             f"Joint: {joint_name:<20} | Joint ID: {joint_id!s:<3} | "
    #             f"Damping: {damping:6.3f} | Armature: {armature:6.3f} | "
    #             f"Friction: {frictionloss:6.3f}"
    #         )

    #         if actuator_id >= 0:
    #             forcerange = actuator_forcerange[actuator_id]
    #             line += (
    #                 f" | Actuator: {actuator_name:<20} (ID: {actuator_id:2d}) | "
    #                 f"Forcerange: [{forcerange[0]:6.3f}, {forcerange[1]:6.3f}] | "
    #                 f"Kp: {kp} | Kd: {kd}"
    #             )
    #         else:
    #             line += " | Actuator: N/A (passive joint)"

    #         debug_lines.append(line)

    #     logger.info("\n".join(debug_lines))
    

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        """Get joint metadata from metadata.json file."""
        metadata_path = Path(self.config.robot_mjcf_path) / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        try:
            with open(metadata_path, "r") as f:
                raw_metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {metadata_path}: {e}")

        if "joint_name_to_metadata" not in raw_metadata:
            raise ValueError(f"'joint_name_to_metadata' key missing in {metadata_path}")

        joint_metadata = raw_metadata["joint_name_to_metadata"]
        if not isinstance(joint_metadata, dict):
            raise TypeError(f"'joint_name_to_metadata' in {metadata_path} must be a dictionary.")

        # Convert raw metadata to JointMetadataOutput objects
        return {joint_name: JointMetadataOutput(**metadata) for joint_name, metadata in joint_metadata.items()}

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        if metadata is None:
            raise ValueError("Metadata is required to get actuators")

        self.joint_mappings = self.create_joint_mappings(physics_model, metadata)

        num_joints = len(self.joint_mappings)

        max_torque_j = jnp.zeros(num_joints)
        max_velocity_j = jnp.zeros(num_joints)
        max_pwm_j = jnp.zeros(num_joints)
        vin_j = jnp.zeros(num_joints)
        kt_j = jnp.zeros(num_joints)
        r_j = jnp.zeros(num_joints)
        # vmax_j = jnp.zeros(num_joints)
        # amax_j = jnp.zeros(num_joints)
        vmax_j = jnp.ones(num_joints) * 5.0  # or whatever test value you want
        amax_j = jnp.ones(num_joints) * 39.0
        kp_j = jnp.zeros(num_joints)
        kd_j = jnp.zeros(num_joints)
        error_gain_j = jnp.zeros(num_joints)

        # Validate parameters
        required_keys = ["max_torque", "error_gain", "max_velocity", "max_pwm", "vin", "kt", "R"]

        # Sort joint_mappings by actuator_id to ensure correct ordering
        sorted_joints = sorted(self.joint_mappings.items(), key=lambda x: x[1]["actuator_id"])

        for i, (joint_name, mapping) in enumerate(sorted_joints):
            joint_metadata = metadata[joint_name]
            if not isinstance(joint_metadata, JointMetadataOutput):
                raise TypeError(f"Metadata entry for joint '{joint_name}' must be a JointMetadataOutput.")

            actuator_type = cast(str, joint_metadata.actuator_type)
            if actuator_type is None:
                raise ValueError(f"'actuator_type' is not available for joint {joint_name}")
            if not isinstance(actuator_type, str):
                raise TypeError(f"'actuator_type' for joint {joint_name} must be a string.")

            params = load_actuator_params(self.config.actuator_params_path, actuator_type)

            # Validate parameters
            for key in required_keys:
                if key not in params:
                    raise ValueError(f"Missing required key '{key}' in {actuator_type} parameters.")

            max_torque_j = max_torque_j.at[i].set(params["max_torque"])
            max_velocity_j = max_velocity_j.at[i].set(params["max_velocity"])
            max_pwm_j = max_pwm_j.at[i].set(params["max_pwm"])
            vin_j = vin_j.at[i].set(params["vin"])
            kt_j = kt_j.at[i].set(params["kt"])
            r_j = r_j.at[i].set(params["R"])
            error_gain_j = error_gain_j.at[i].set(params["error_gain"])

            # Set kp and kd values
            if joint_metadata.kp is None or joint_metadata.kd is None:
                raise ValueError(f"kp/kd values for joint {joint_name} are not available")
            kp_j = kp_j.at[i].set(float(joint_metadata.kp))
            kd_j = kd_j.at[i].set(float(joint_metadata.kd))

        # self.log_joint_config(physics_model)
        log_joint_config(physics_model, metadata)

        return FeetechActuators(
            max_torque_j=max_torque_j,
            max_velocity_j=max_velocity_j,
            max_pwm_j=max_pwm_j,
            vin_j=vin_j,
            kt_j=kt_j,
            r_j=r_j,
            kp_j=kp_j,
            kd_j=kd_j,
            vmax_j=vmax_j,
            amax_j=amax_j,
            dt=self.config.dt,
            error_gain_j=error_gain_j,
            action_noise=0.0,
            action_noise_type="none",
            torque_noise=0.0,
            torque_noise_type="none",
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.FloorFrictionRandomizer.from_geom_name(physics_model, "floor", scale_lower=0.8, scale_upper=1.2),
            ksim.ArmatureRandomizer(),
            ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.95, scale_upper=1.05),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-2), scale_upper=math.radians(2)),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                x_force=1.5,
                y_force=1.5,
                z_force=0.1,
                x_angular_force=0.1,
                y_angular_force=0.1,
                z_angular_force=0.3,
                interval_range=(0.5, 4.0),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in ZEROS}, scale=0.1),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(),
            ksim.JointVelocityObservation(),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="base_link_quat",
                lag_range=(0.0, 0.5),
            ),
            ksim.ActuatorAccelerationObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro"),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return []

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            # Standard rewards.
            ksim.StayAliveReward(scale=1.0),
            ksim.NaiveForwardReward(clip_min=0.0, clip_max=0.5, scale=1.0),
            ksim.UprightReward(index="z", inverted=False, scale=0.1),
            # # Normalization penalties.
            # ksim.ActionInBoundsReward.create(physics_model, scale=0.01),
            # ksim.ActionSmoothnessPenalty(scale=-0.01),
            # ksim.ActuatorJerkPenalty(ctrl_dt=self.config.ctrl_dt, scale=-0.001),
            # ksim.ActuatorRelativeForcePenalty.create(physics_model, scale=-0.001),
            # ksim.AngularVelocityPenalty(index="x", scale=-0.0005),
            # ksim.AngularVelocityPenalty(index="y", scale=-0.0005),
            # ksim.AngularVelocityPenalty(index="z", scale=-0.0005),
            # ksim.LinearVelocityPenalty(index="y", scale=-0.0005),
            # ksim.LinearVelocityPenalty(index="z", scale=-0.0005),
            # # Bespoke rewards.
            # BentArmPenalty.create(physics_model, scale=-0.01),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.05, unhealthy_z_upper=0.5),
            # ksim.PitchTooGreatTermination(max_pitch=math.radians(30)),
            # ksim.RollTooGreatTermination(max_roll=math.radians(30)),
            # ksim.HighVelocityTermination(),
            # ksim.FarFromOriginTermination(max_dist=10.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.EpisodeLengthCurriculum(
            num_levels=self.config.num_curriculum_levels,
            increase_threshold=self.config.increase_threshold,
            decrease_threshold=self.config.decrease_threshold,
            min_level_steps=self.config.min_level_steps,
            dt=self.config.ctrl_dt,
            min_level=self.config.min_curriculum_level,
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        return Model(
            key,
            num_inputs=NUM_ACTOR_INPUTS,
            num_outputs=NUM_JOINTS,
            min_std=0.01,
            max_std=1.0,
            hidden_size=self.config.hidden_size,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]

        obs_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                proj_grav_3,  # 3
            ],
            axis=-1,
        )

        action, carry = model.forward(obs_n, carry)

        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]

        obs_n = jnp.concatenate(
            [
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        def scan_fn(
            actor_critic_carry: tuple[Array, Array],
            transition: ksim.Trajectory,
        ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
            actor_carry, critic_carry = actor_critic_carry
            actor_dist, next_actor_carry = self.run_actor(
                model=model.actor,
                observations=transition.obs,
                commands=transition.command,
                carry=actor_carry,
            )
            log_probs = actor_dist.log_prob(transition.action)
            assert isinstance(log_probs, Array)
            value, next_critic_carry = self.run_critic(
                model=model.critic,
                observations=transition.obs,
                commands=transition.command,
                carry=critic_carry,
            )

            transition_ppo_variables = ksim.PPOVariables(
                log_probs=log_probs,
                values=value.squeeze(-1),
            )

            next_carry = jax.tree.map(
                lambda x, y: jnp.where(transition.done, x, y),
                self.get_initial_model_carry(rng),
                (next_actor_carry, next_critic_carry),
            )

            return next_carry, transition_ppo_variables

        next_model_carry, ppo_variables = jax.lax.scan(scan_fn, model_carry, trajectory)

        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        return ksim.Action(
            action=action_j,
            carry=(actor_carry, critic_carry_in),
            aux_outputs=None,
        )

    def create_joint_mappings(
        self, physics_model: ksim.PhysicsModel, metadata: dict[str, JointMetadataOutput]
    ) -> dict[str, dict]:
        """Creates mappings between joint names, nn_ids, and actuator_ids.

        Args:
            physics_model: The MuJoCo/MJX model containing joint information
            metadata: The joint metadata dictionary from metadata.json

        Returns:
            Dictionary mapping joint names to their nn_id and actuator_id
        """
        debug_lines = ["==== Joint Name to ID Mappings ===="]

        # Get ordered list of joints from MuJoCo/MJX model
        if isinstance(physics_model, mujoco.MjModel):
            mujoco_joints = [
                mujoco.mj_id2name(physics_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                for i in range(physics_model.njnt)
                if mujoco.mj_id2name(physics_model, mujoco.mjtObj.mjOBJ_JOINT, i) is not None
            ]
        else:  # MJX model

            def extract_joint_name(model: mjx.Model, idx: int) -> Optional[str]:
                adr = model.name_jntadr[idx]
                if adr < 0:
                    return None
                end = model.names.find(b"\x00", adr)
                return model.names[adr:end].decode("utf-8")

            mujoco_joints = [
                name for i in range(physics_model.njnt) if (name := extract_joint_name(physics_model, i)) is not None
            ]

        # Create mappings using joint names as keys
        joint_mappings = {}

        # Map each joint, using MuJoCo order for nn_ids
        for nn_id, joint_name in enumerate(mujoco_joints):
            if joint_name in metadata:
                actuator_id = metadata[joint_name].id
                if actuator_id is None:
                    logger.warning("Joint %s has no actuator id", joint_name)
                joint_mappings[joint_name] = {"nn_id": nn_id, "actuator_id": actuator_id}

                debug_lines.append("%-30s -> nn_id: %2d, actuator_id: %s" % (joint_name, nn_id, str(actuator_id)))
            else:
                logger.warning("Joint %s not found in metadata", joint_name)

        logger.info("\n".join(debug_lines))
        return joint_mappings


if __name__ == "__main__":
    ZbotWalkingTask.launch(
        ZbotWalkingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=2,
            epochs_per_log_step=1,
            rollout_length_seconds=8.0,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            max_action_latency=0.01,
            # Checkpointing parameters.
            save_every_n_seconds=60,
            valid_every_n_steps=2,
            valid_first_n_steps=1,
        ),
    )
