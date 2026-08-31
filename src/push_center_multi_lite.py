"""
Reconstruction of the VP2 benchmark's RoboSuite pushing task
(PushCenterMulti) that the iVideoGPT checkpoint was fine-tuned on, rebuilt on
modern robosuite/mujoco. The original fork requires legacy free-mujoco-py and
iGibson rendering, neither viable here (no Apple Silicon support).

Object sizes/densities/frictions, table geometry, placement ranges, and
camera pose are ported verbatim from the pinned fork
(s-tian/robosuite @ 163e4d4, push_center_multi.py); source lines are noted
inline. This reproduces exact geometry and camera projection but not
iGibson's PBR shading, which is a known appearance-domain shift discussed in
the paper's Limitations section.
"""
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

# Exact camera pose for "agentview_shift_2", ported verbatim from
# robosuite_fork/robosuite/models/assets/arenas/table_arena.xml:52 (pinned
# commit 163e4d4). MuJoCo camera quat convention: (w, x, y, z).
AGENTVIEW_SHIFT_2_POS = (0.6758858572576728, -4.760939554065248e-08, 1.251269639768581)
AGENTVIEW_SHIFT_2_QUAT = (0.6325998902320862, 0.31593844294548035, 0.31593817472457886, 0.6326003670692444)


class PushCenterMultiLite(ManipulationEnv):
    """Geometry/camera-faithful reconstruction of PushCenterMulti for calibration only.

    Object set, sizes, densities, frictions, table size/offset, and placement
    ranges are ported verbatim from push_center_multi.py's `_load_model`
    (fork lines 409-486). Reward/task-success logic is intentionally trivial
    (not used for calibration, which never calls sim.step()).
    """

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),  # push_center_multi.py:143
        table_friction=(1.0, 5e-3, 1e-4),  # push_center_multi.py:144
        use_camera_obs=True,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview_shift_2",
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
        hard_reset=True,
        camera_names="agentview_shift_2",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))  # push_center_multi.py:173
        self.use_object_obs = use_object_obs
        self.placement_initializer = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(
            self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        )

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        # Add the exact VP2 training camera (not a stock robosuite camera name).
        mujoco_arena.set_camera(
            camera_name="agentview_shift_2", pos=AGENTVIEW_SHIFT_2_POS, quat=AGENTVIEW_SHIFT_2_QUAT
        )

        # Objects verbatim from push_center_multi.py:409-465, with plain rgba
        # colors in place of the fork's unavailable texture pack.
        self.box_object = BoxObject(
            name="cube", size=[0.06, 0.06, 0.06], density=1000,
            rgba=[0.8, 0.1, 0.1, 1], friction=[0.7, 0.005, 0.0001],
        )
        self.box_object2 = BoxObject(
            name="cube2", size=[0.04, 0.04, 0.04], density=1000,
            rgba=[0.1, 0.1, 0.8, 1], friction=[0.7, 0.005, 0.0001],
        )
        self.ball_object = BallObject(
            name="ball", size=[0.07], density=1000,
            rgba=[0.1, 0.8, 0.1, 1], friction=[0.7, 0.005, 0.0001],
        )
        self.cylinder_object = CylinderObject(
            name="cylinder", size=[0.06, 0.06], density=1000,
            rgba=[0.8, 0.8, 0.1, 1], friction=[0.7, 0.005, 0.0001],
        )
        self.objects = [self.box_object, self.box_object2, self.cylinder_object, self.ball_object]

        # Placement ranges verbatim from push_center_multi.py:476-486.
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.objects,
            x_range=[-0.1, 0.3],
            y_range=[-0.15, 0.15],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    def _setup_references(self):
        super()._setup_references()
        self.object_body_ids = [self.sim.model.body_name2id(obj.root_body) for obj in self.objects]

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
