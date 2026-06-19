#!/usr/bin/env python3
"""
Environment: builds the full MuJoCo scene in Python (no scene.xml needed).

Loading pipeline:
  1. Parse the robot XML with ElementTree.
  2. Append free-floating objects and static obstacles to <worldbody>.
  3. Compile via mujoco.MjModel.from_xml_string() with absolute mesh paths.
  4. Wrap in Robot for arm/gripper control.

Adding scene content:
  Pass `objects` / `obstacles` dicts to `environment(...)` — see the schemas
  on OBJECT_DEFAULTS and OBSTACLE_DEFAULTS below for accepted fields.
"""
import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

from robot import Robot

# ── Default scene content ─────────────────────────────────────────────────────
#
# OBJECT_DEFAULTS schema:
#   name -> {
#       "pos":  [x, y, z],                    required
#       "quat": [w, x, y, z],                 default [1, 0, 0, 0]
#       "size": [hx, hy, hz] | [r],           default [0.03, 0.03, 0.03]
#       "mass": float                         default 0.1
#       "rgba": [r, g, b, a]                  default [0.8, 0.2, 0.2, 1]
#       "type": "box" | "sphere" | "cylinder" default "box"
#   }
#
# OBSTACLE_DEFAULTS schema (list of dicts):
#   {
#       "name": str                           required
#       "pos":  [x, y, z]                     required
#       "size": [hx, hy, hz] | [r] | [r, h]   required
#       "rgba": [r, g, b, a]                  default [0.9, 0.5, 0.1, 1]
#       "type": "box" | "sphere" | "cylinder" default "box"
#   }

OBJECT_DEFAULTS = {}

PDDL_PROBLEM_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "problem_chem_simplified_no_redundant_predicates.pddl",
))

# Externally chosen cylinder radii, in meters. The PDDL gives x/y centers and
# total z height; the radius is a scene-design choice.
PDDL_CYLINDER_RADII = {
    "in-2": 0.050,
    "in-4": 0.055,
    "in-5": 0.047,
    "out-2": 0.060,
}

PDDL_CYLINDER_COLORS = {
    "in-2":  [0.95, 0.68, 0.20, 0.85],
    "in-4":  [0.20, 0.75, 0.90, 0.85],
    "in-5":  [0.95, 0.45, 0.20, 0.85],
    "out-2": [0.88, 0.88, 0.24, 0.85],
}

REMOVED_PDDL_CYLINDERS = {"in-1", "in-3", "out-1"}

PDDL_LAYOUT_ROTATION_DEG = -90.0

_PDDL_BUCKET_RE = re.compile(
    r"\(=\s+\(bucket-([xyz])\s+([^\s)]+)\)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\)"
)


def _read_pddl_bucket_xyz(problem_path):
    """Return {bucket_name: (x, y, z_height)} parsed from a PDDL problem."""
    try:
        with open(problem_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return {}

    coords = {}
    for axis, name, value in _PDDL_BUCKET_RE.findall(text):
        coords.setdefault(name, {})[axis] = float(value)

    return {
        name: (values["x"], values["y"], values["z"])
        for name, values in coords.items()
        if {"x", "y", "z"}.issubset(values)
    }


def _rotate_xy(x, y, cx, cy, degrees):
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = x - cx, y - cy
    return cx + c * dx - s * dy, cy + s * dx + c * dy


def _bucket_layout_center(bucket_coords):
    if not bucket_coords:
        return 0.0, 0.0
    center_x = sum(x for x, _, _ in bucket_coords.values()) / len(bucket_coords)
    center_y = sum(y for _, y, _ in bucket_coords.values()) / len(bucket_coords)
    return center_x, center_y


def load_pddl_bucket_targets(problem_path=PDDL_PROBLEM_PATH,
                             rotate_layout=True,
                             z_clearance=0.0):
    """Return scene-frame EE targets for PDDL beakers keyed by beaker name."""
    bucket_coords = _read_pddl_bucket_xyz(problem_path)
    center_x, center_y = _bucket_layout_center(bucket_coords)

    targets = {}
    for name, (x, y, z) in bucket_coords.items():
        if rotate_layout:
            x, y = _rotate_xy(x, y, center_x, center_y,
                              PDDL_LAYOUT_ROTATION_DEG)
        targets[name] = np.array(
            [float(x), float(y), float(z) + float(z_clearance)],
            dtype=np.float64,
        )
    return targets


def load_pddl_cylinder_obstacles(problem_path=PDDL_PROBLEM_PATH):
    """Build static cylinder obstacles from PDDL bucket coordinates."""
    obstacles = []
    bucket_coords = _read_pddl_bucket_xyz(problem_path)
    center_x, center_y = _bucket_layout_center(bucket_coords)

    for index, (name, (x, y, height)) in enumerate(bucket_coords.items()):
        if name in REMOVED_PDDL_CYLINDERS:
            continue
        if height <= 0.0:
            continue

        fallback_radius = 0.040 + 0.004 * (index % 6)
        radius = PDDL_CYLINDER_RADII.get(name, fallback_radius)
        half_height = 0.5 * height
        rotated_x, rotated_y = _rotate_xy(
            x, y, center_x, center_y, PDDL_LAYOUT_ROTATION_DEG)

        obstacles.append({
            "name": f"pddl_{name.replace('-', '_')}_cylinder",
            "pos": [float(rotated_x), float(rotated_y), half_height],
            "size": [radius, half_height],
            "rgba": PDDL_CYLINDER_COLORS.get(name, [0.9, 0.5, 0.1, 0.85]),
            "type": "cylinder",
            "contype": 1,
            "conaffinity": 1,
        })
    return obstacles


OBSTACLE_DEFAULTS = load_pddl_cylinder_obstacles()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _solid_inertia(shape, size, mass):
    """Diagonal inertia (Ixx, Iyy, Izz) of a uniform solid primitive at COM.

    box      : size = [hx, hy, hz] half-extents
    sphere   : size = [r]
    cylinder : size = [r, half-height]  (axis = z)
    """
    if shape == "box":
        a, b, c = size[0], size[1], size[2]
        ix = mass * (b * b + c * c) / 3.0
        iy = mass * (a * a + c * c) / 3.0
        iz = mass * (a * a + b * b) / 3.0
    elif shape == "sphere":
        r  = size[0]
        ix = iy = iz = 0.4 * mass * r * r
    elif shape == "cylinder":
        r, h = size[0], size[1]
        ix = iy = mass * (3.0 * r * r + (2.0 * h) ** 2) / 12.0
        iz = 0.5 * mass * r * r
    else:
        raise ValueError(f"unsupported shape {shape!r}")
    return ix, iy, iz


def _fmt(vec):
    return " ".join(repr(float(v)) for v in vec)


def _add_free_object(worldbody, name, spec):
    """Append a free-floating body driven by a freejoint."""
    pos   = spec["pos"]
    quat  = spec.get("quat", [1.0, 0.0, 0.0, 0.0])
    shape = spec.get("type", "box")
    size  = spec.get("size", [0.03, 0.03, 0.03])
    mass  = float(spec.get("mass", 0.1))
    rgba  = spec.get("rgba", [0.8, 0.2, 0.2, 1.0])

    b = ET.SubElement(worldbody, 'body',
                      name=name,
                      pos=_fmt(pos),
                      quat=_fmt(quat))
    ET.SubElement(b, 'freejoint', name=f'{name}_free')

    ix, iy, iz = _solid_inertia(shape, size, mass)
    ET.SubElement(b, 'inertial',
                  mass=str(mass), pos='0 0 0',
                  diaginertia=f'{ix} {iy} {iz}')
    geom_attrs = {
        "name": f"{name}_geom",
        "type": shape,
        "size": _fmt(size),
        "rgba": _fmt(rgba),
    }
    for key in ("contype", "conaffinity", "condim", "margin", "gap", "friction"):
        if key in spec:
            value = spec[key]
            geom_attrs[key] = _fmt(value) if isinstance(value, (list, tuple)) else str(value)
    ET.SubElement(b, 'geom', **geom_attrs)


def _add_static_obstacle(worldbody, spec):
    """Append a static body (no joint) attached to the world."""
    name  = spec["name"]
    pos   = spec["pos"]
    shape = spec.get("type", "box")
    size  = spec["size"]
    rgba  = spec.get("rgba", [0.9, 0.5, 0.1, 1.0])

    b = ET.SubElement(worldbody, 'body', name=name, pos=_fmt(pos))
    geom_attrs = {
        "name": f"{name}_geom",
        "type": shape,
        "size": _fmt(size),
        "rgba": _fmt(rgba),
    }
    for key in ("contype", "conaffinity", "condim", "margin", "gap"):
        if key in spec:
            geom_attrs[key] = str(spec[key])
    ET.SubElement(b, 'geom', **geom_attrs)


# ── Scene builder ──────────────────────────────────────────────────────────────

def _build_model(robot_xml_path, objects, obstacles):
    """Load the robot XML, add scene content, compile, and return MjModel."""
    abs_xml  = os.path.abspath(robot_xml_path)
    base_dir = os.path.dirname(abs_xml)

    tree = ET.parse(abs_xml)
    root = tree.getroot()

    # mj_compile loses the original file path, so relative mesh paths fail
    # unless meshdir is rewritten to an absolute one.
    compiler = root.find('compiler')
    rel_meshdir = compiler.get('meshdir', '')
    abs_meshdir = os.path.join(base_dir, rel_meshdir) if rel_meshdir else base_dir
    compiler.set('meshdir', abs_meshdir)

    worldbody = root.find('worldbody')

    # Append objects after the robot so freejoint qpos slots come AFTER the
    # arm joints. Order: arm joints, gripper, then one freejoint per object.
    for name, spec in objects.items():
        _add_free_object(worldbody, name, spec)
    for spec in obstacles:
        _add_static_obstacle(worldbody, spec)

    xml_str = ET.tostring(root, encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml_str)


# ── Environment class ──────────────────────────────────────────────────────────

class environment:
    """
    Wraps Robot + a Python-built scene to provide object management.

    Usage:
        env = environment(
            "ur10e",
            objects={"cube": {"pos": [0.4, 0, 0.5]}},
            obstacles=[{"name": "wall", "pos": [-0.5, 0, 0.4], "size": [0.02, 0.4, 0.4]}],
        )
        env.reset()

        pos  = env.get_object_pos("cube")
        quat = env.get_object_quat("cube")     # [w, x, y, z]
        env.set_object_pose("cube", pos=[0.5, 0.1, 0.57])

        tau = controller.compute()
        env.step(tau)
    """

    def __init__(self, robot="ur10e", objects=None, obstacles=None):
        from robot_config import get_config
        cfg = get_config(robot)

        # Take a defensive copy so callers can mutate their dicts later.
        self._object_defs   = dict(objects   if objects   is not None else OBJECT_DEFAULTS)
        self._obstacle_defs = list(obstacles if obstacles is not None else OBSTACLE_DEFAULTS)

        model = _build_model(cfg["xml"], self._object_defs, self._obstacle_defs)
        self.robot = Robot(robot, model=model)

        # Pre-cache body IDs and freejoint qpos/qvel addresses.
        # model.jnt_qposadr[id] → start index in data.qpos
        # model.jnt_dofadr[id]  → start index in data.qvel
        self._objects = {}
        for name in self._object_defs:
            body_id = mujoco.mj_name2id(
                self.robot.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jnt_id  = mujoco.mj_name2id(
                self.robot.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            self._objects[name] = {
                "body_id": body_id,
                "qadr":    self.robot.model.jnt_qposadr[jnt_id],
                "vadr":    self.robot.model.jnt_dofadr[jnt_id],
            }

        self._obstacles = {}
        for spec in self._obstacle_defs:
            name = spec["name"]
            self._obstacles[name] = mujoco.mj_name2id(
                self.robot.model, mujoco.mjtObj.mjOBJ_BODY, name)

    # ── Scene control ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset arm to home and all objects to their default poses."""
        self.robot.reset()
        for name, spec in self._object_defs.items():
            self.set_object_pose(name,
                                 spec["pos"],
                                 spec.get("quat", [1.0, 0.0, 0.0, 0.0]))
        mujoco.mj_forward(self.robot.model, self.robot.data)

    def step(self, tau=None):
        """Advance one simulation step (delegates to robot.step)."""
        self.robot.step(tau)

    # ── Object state ───────────────────────────────────────────────────────────

    def get_object_pos(self, name):
        """World position (3,) of object 'name'."""
        bid = self._objects[name]["body_id"]
        return self.robot.data.xpos[bid].copy()

    def get_object_quat(self, name):
        """World orientation [w, x, y, z] of object 'name'."""
        bid = self._objects[name]["body_id"]
        return self.robot.data.xquat[bid].copy()

    def get_object_pose(self, name):
        """SE3 (4,4) of object 'name' in world frame."""
        T = np.eye(4)
        bid = self._objects[name]["body_id"]
        T[:3, :3] = self.robot.data.xmat[bid].reshape(3, 3)
        T[:3,  3] = self.robot.data.xpos[bid]
        return T

    def set_object_pose(self, name, pos, quat=None):
        """Teleport object to (pos, quat) and zero its velocity."""
        obj = self._objects[name]
        qadr, vadr = obj["qadr"], obj["vadr"]
        self.robot.data.qpos[qadr:qadr + 3] = pos
        if quat is not None:
            self.robot.data.qpos[qadr + 3:qadr + 7] = quat
        self.robot.data.qvel[vadr:vadr + 6] = 0.0

    # ── Convenience ────────────────────────────────────────────────────────────

    @property
    def object_names(self):
        """List of all managed object names."""
        return list(self._objects.keys())

    @property
    def obstacle_names(self):
        """List of all static obstacle names."""
        return list(self._obstacles.keys())
