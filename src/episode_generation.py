"""
Episode generation and per-timestep logging on the PushCenterMultiLite
environment: a scripted multi-segment approach-push-retract policy plus
contact/state logging, used to produce every episode batch (exploratory
seeds 0-9, confirmation seeds 100-109).

Scripted policy: per episode, run several (default 3) independent
approach -> push -> retract cycles targeting the cube object, each with a
randomized push direction and randomized approach offset, so one episode
naturally contains multiple separate contact interaction segments rather than
one long sustained contact.

Uses OSC_POSITION proportional control: action_xyz = clip((target-eef)/0.05, -1, 1).
Gripper action held closed (constant) throughout -- this is a pushing task,
not grasping.
"""
import numpy as np

from . import simulator_interface as lib

OBJ_INDEX = 0  # "cube"
STEP_SIZE_M = 0.05  # matches controller output_max/min


def gripper_contact_geoms(env):
    return set(env.robots[0].gripper["right"].contact_geoms)


def object_contact_geoms(env):
    return {i: set(obj.contact_geoms) for i, obj in enumerate(env.objects)}


def classify_contacts(env, grip_geoms, obj_geoms):
    """Returns dict: obj_index -> True/False (gripper touching that object
    this timestep), by scanning sim.data.contact[0:ncon] and resolving geom
    ids to names via sim.model.geom_id2name -- per-object, not a single
    global contact boolean.
    """
    result = {i: False for i in obj_geoms}
    ncon = env.sim.data.ncon
    for i in range(ncon):
        c = env.sim.data.contact[i]
        g1 = env.sim.model.geom_id2name(c.geom1)
        g2 = env.sim.model.geom_id2name(c.geom2)
        names = {g1, g2}
        touches_gripper = len(names & grip_geoms) > 0
        if not touches_gripper:
            continue
        for obj_idx, ogeoms in obj_geoms.items():
            if len(names & ogeoms) > 0:
                result[obj_idx] = True
    return result


def eef_pos(env):
    return env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]].copy()


def proportional_action(target_xyz, current_xyz, gripper_closed=True):
    delta = (target_xyz - current_xyz) / STEP_SIZE_M
    delta = np.clip(delta, -1.0, 1.0)
    grip = np.array([1.0 if gripper_closed else -1.0])
    return np.concatenate([delta, grip]).astype(np.float32)


def run_episode(seed, n_cycles=3, hover_h=0.12, approach_steps=6, push_steps=5, retract_steps=4, rng=None,
                 obj_index=OBJ_INDEX):
    """Runs one episode with n_cycles approach-push-retract segments against
    the manipulated object (default: the cube, OBJ_INDEX=0; pass obj_index to
    target a different object, e.g. 3="ball" -- see push_center_multi_lite.py's
    self.objects list order), randomized push direction/approach offset per
    cycle and randomized initial object placement (via env.reset()). Logs
    everything the task asks for at every timestep, for ALL objects
    (contact_per_object, obj_pos, etc. are unaffected by obj_index -- only
    which object the scripted policy chases changes).

    obj_index is the ONLY behavioral change from the frozen cube pipeline:
    every geometric constant below (hover height, approach/push offsets,
    descend height) is applied identically regardless of which object's
    position is substituted in. Default value reproduces the original cube
    behavior exactly.

    Returns a dict of per-timestep arrays/lists ready to be sliced into
    windows for the natural rollout-error analysis.
    """
    rng = rng or np.random.default_rng(seed)
    env = lib.build_env(seed=seed)

    grip_geoms = gripper_contact_geoms(env)
    obj_geoms = object_contact_geoms(env)

    log = {
        "raw_frame_256": [], "model_frame_64": [], "action": [],
        "eef_pos": [], "obj_pos": [], "obj_quat": [],
        "contact_per_object": [], "qpos": [], "qvel": [], "time": [],
        "action_norm": [], "obj_velocity": [],
    }

    def record(action_taken):
        img = lib.render_raw(env)
        frame64 = lib.preprocess_to_model_input(img).numpy()
        contacts = classify_contacts(env, grip_geoms, obj_geoms)
        obj_positions = [lib.get_object_world_pos(env, i) for i in range(len(env.objects))]
        obj_quats = [env.sim.data.body_xquat[env.object_body_ids[i]].copy() for i in range(len(env.objects))]
        obj_vel = [np.linalg.norm(env.sim.data.get_body_xvelp(env.objects[i].root_body)) for i in range(len(env.objects))]
        sim_state = env.sim.get_state()
        log["raw_frame_256"].append(img)
        log["model_frame_64"].append(frame64)
        log["action"].append(action_taken.copy())
        log["eef_pos"].append(eef_pos(env))
        log["obj_pos"].append(np.stack(obj_positions))
        log["obj_quat"].append(np.stack(obj_quats))
        log["contact_per_object"].append([contacts[i] for i in range(len(env.objects))])
        log["qpos"].append(sim_state.qpos.copy())
        log["qvel"].append(sim_state.qvel.copy())
        log["time"].append(sim_state.time)
        log["action_norm"].append(float(np.linalg.norm(action_taken[:3])))
        log["obj_velocity"].append(obj_vel)

    # record the initial (post-reset) state as t=0 with a zero action
    record(np.zeros(4, dtype=np.float32))

    for cycle in range(n_cycles):
        cube_pos = lib.get_object_world_pos(env, obj_index)
        angle = rng.uniform(0, 2 * np.pi)
        push_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
        approach_pos = cube_pos - push_dir * rng.uniform(0.10, 0.16) + np.array([0, 0, hover_h])

        for _ in range(approach_steps):
            a = proportional_action(approach_pos, eef_pos(env))
            env.step(a)
            record(a)

        descend_pos = approach_pos.copy()
        descend_pos[2] = cube_pos[2] + 0.01
        for _ in range(2):
            a = proportional_action(descend_pos, eef_pos(env))
            env.step(a)
            record(a)

        push_target = cube_pos + push_dir * rng.uniform(0.10, 0.18)
        push_target[2] = descend_pos[2]
        for _ in range(push_steps):
            a = proportional_action(push_target, eef_pos(env))
            env.step(a)
            record(a)

        retract_pos = eef_pos(env) + np.array([0, 0, hover_h * 1.5])
        for _ in range(retract_steps):
            a = proportional_action(retract_pos, eef_pos(env))
            env.step(a)
            record(a)

    for k in log:
        log[k] = np.array(log[k])
    return log


def find_contact_events(contact_per_object, obj_index=0):
    """contact_per_object: (T, n_objects) bool array. Returns onset and
    release timestep indices for obj_index, where an "event" is a
    False->True (onset) or True->False (release) transition -- sustained
    contact across consecutive timesteps is one event, not many.
    """
    col = contact_per_object[:, obj_index].astype(bool)
    onsets = [t for t in range(1, len(col)) if col[t] and not col[t - 1]]
    releases = [t for t in range(1, len(col)) if not col[t] and col[t - 1]]
    if col[0]:
        onsets = [0] + onsets
    return onsets, releases
